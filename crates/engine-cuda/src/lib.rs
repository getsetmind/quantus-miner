#![deny(rust_2018_idioms)]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, sys as nvrtc_sys, CompileOptions, Ptx};
use engine_cpu::{CancelCheck, Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use pow_core::{format_hashrate, format_u512, JobContext};
use primitive_types::U512;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const KERNEL_SRC: &str = include_str!("kernels/mining.cu");
const THREADS_PER_BLOCK: u32 = 256;
const MAX_BLOCKS: u32 = 4096;
/// Candidate count, then up to `MAX_HITS` candidate indices in claim order.
const MAX_HITS: usize = 8;
const RESULTS_U32S: usize = 1 + MAX_HITS;

#[repr(C)]
#[derive(Clone, Copy)]
struct MiningParams {
    prestate: [u32; 24],
    start_nonce: [u32; 16],
    difficulty_target: [u32; 16],
    dispatch_config: [u32; 3],
}

unsafe impl cudarc::driver::DeviceRepr for MiningParams {}

struct CudaDevice {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    name: String,
}

struct WorkerBuffers {
    engine_id: usize,
    device_index: usize,
    stream: Arc<CudaStream>,
    results: CudaSlice<u32>,
    midstate: CudaSlice<u32>,
    start_nonce: CudaSlice<u32>,
    hashes: CudaSlice<u32>,
    mine: CudaFunction,
    hash: CudaFunction,
    busy: Duration,
    busy_since: Instant,
}

/// Native CUDA mining engine.
///
/// Search contract: every launch evaluates its whole nonce rectangle; no
/// thread stops early. Up to `MAX_HITS` candidates per launch are recorded
/// in atomic claim order. If that buffer overflows, the same start is replayed
/// with a smaller batch until all candidates fit. The host recomputes each one
/// with the exact CPU hash, returning the lowest that is really below the target.
/// The kernel's Goldilocks reduction skips two carry corrections (`reduce128`
/// in `kernels/mining.cu`): each of the roughly 1,470 field multiplies per
/// hash can be off by EPS mod p with probability about 2^-33, so about one
/// nonce in three million is hashed wrong on the GPU. A wrong hash that lands
/// below the target is rejected by the CPU check; a wrong hash for a nonce that
/// is actually valid is missed and the range reports `Exhausted`. The expected
/// loss is about 3e-7 of solutions, far below the throughput the shortcut buys.
/// `hash_count` is the number of nonces the launches evaluated, including
/// overflow replays and wrong hashes. `hash_nonces` is subject to the same
/// contract: it is a kernel self-test, not a verifier; use
/// `pow_core::hash_from_nonce`.
pub struct CudaEngine {
    engine_id: usize,
    devices: Vec<Arc<CudaDevice>>,
    device_counter: AtomicUsize,
    batch_size: u32,
    throttle_ms: u64,
}

static ENGINE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ASSIGNED_DEVICE: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static WORKER_BUFFERS: RefCell<Option<WorkerBuffers>> = const { RefCell::new(None) };
    static DEVICE_LOST: RefCell<Option<usize>> = const { RefCell::new(None) };
}

fn select_ordinals(
    count: usize,
    ordinals: Option<&[usize]>,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let selected = ordinals.map_or_else(|| (0..count).collect::<Vec<_>>(), <[usize]>::to_vec);
    if selected.is_empty()
        || selected.len() > count
        || selected.iter().any(|&n| n >= count)
        || selected
            .iter()
            .enumerate()
            .any(|(i, n)| selected[..i].contains(n))
    {
        return Err("CUDA device selection must contain distinct valid visible ordinals".into());
    }
    Ok(selected)
}

impl CudaEngine {
    pub fn try_new(batch_size: u32, throttle_ms: u64) -> Result<Self, Box<dyn std::error::Error>> {
        Self::try_new_selected(batch_size, throttle_ms, None)
    }

    /// Select CUDA-visible ordinals explicitly, without initializing other GPUs.
    pub fn try_new_on_devices(
        batch_size: u32,
        throttle_ms: u64,
        ordinals: &[usize],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::try_new_selected(batch_size, throttle_ms, Some(ordinals))
    }

    fn try_new_selected(
        batch_size: u32,
        throttle_ms: u64,
        ordinals: Option<&[usize]>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if batch_size == 0 {
            return Err("batch_size must be non-zero".into());
        }

        match silent_catch(cudarc::driver::result::init) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(format!(
                    "CUDA driver is not available (need libcuda / nvidia driver): {e}"
                )
                .into());
            }
            Err(_) => {
                return Err("CUDA driver is not available (need libcuda / nvidia driver)".into());
            }
        }
        let count = cudarc::driver::result::device::get_count()
            .map_err(|e| format!("Failed to query CUDA device count: {e}"))?;
        if count <= 0 {
            return Err("No CUDA devices found".into());
        }

        let selected = select_ordinals(count as usize, ordinals)?;
        let mut devices = Vec::new();
        let mut compiled: HashMap<(i32, i32), Ptx> = HashMap::new();
        for ordinal in selected {
            let ctx = CudaContext::new(ordinal)
                .map_err(|e| format!("Failed to create CUDA context for device {ordinal}: {e}"))?;
            ctx.set_blocking_synchronize().map_err(|e| {
                format!("Failed to enable blocking CUDA synchronization on device {ordinal}: {e}")
            })?;
            let name = ctx
                .name()
                .unwrap_or_else(|_| format!("cuda-device-{ordinal}"));
            let (major, minor) = ctx.compute_capability().map_err(|e| {
                format!("Failed to query compute capability of CUDA device {ordinal}: {e}")
            })?;
            let ptx = match compiled.get(&(major, minor)) {
                Some(ptx) => ptx.clone(),
                None => {
                    let ptx = match silent_catch(|| compile_kernel(major, minor)) {
                        Ok(ptx) => ptx?,
                        Err(_) => {
                            return Err("CUDA NVRTC library is not available (need libnvrtc from the CUDA toolkit)".into());
                        }
                    };
                    compiled.insert((major, minor), ptx.clone());
                    ptx
                }
            };
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("Failed to load CUDA mining module on {name}: {e}"))?;
            let mine = module
                .load_function("mining_main")
                .map_err(|e| format!("Failed to load mining_main on {name}: {e}"))?;
            let (regs, local, shared, constant) = (
                mine.num_regs()?,
                mine.local_size_bytes()?,
                mine.shared_size_bytes()?,
                mine.const_size_bytes()?,
            );
            log::info!(
                target: "cuda_engine",
                "CUDA device {ordinal}: {name} (sm_{major}{minor}; mining_main uses {regs} registers, {local} B local, {shared} B shared, {constant} B constant)"
            );
            if local > 0 {
                log::warn!(
                    target: "cuda_engine",
                    "mining_main spills {local} bytes per thread to local memory on {name}"
                );
            }
            devices.push(Arc::new(CudaDevice { ctx, module, name }));
        }

        log::info!(
            target: "cuda_engine",
            "CUDA engine initialized with {} device(s) (batch size: {batch_size} nonces, throttle: {throttle_ms}ms)",
            devices.len()
        );

        Ok(Self {
            engine_id: ENGINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            devices,
            device_counter: AtomicUsize::new(0),
            batch_size,
            throttle_ms,
        })
    }

    /// One independently identified engine per selected device. Each pool worker
    /// owns one engine, so scheduling order cannot alter its device assignment.
    pub fn into_device_engines(self) -> Vec<Self> {
        self.devices
            .into_iter()
            .map(|device| Self {
                engine_id: ENGINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
                devices: vec![device],
                device_counter: AtomicUsize::new(0),
                batch_size: self.batch_size,
                throttle_ms: self.throttle_ms,
            })
            .collect()
    }

    /// Stable device UUID for read-only telemetry matching. A multi-device
    /// engine cannot supply one UUID; split it into explicit device engines.
    pub fn device_uuid(&self) -> Result<String, Box<dyn std::error::Error>> {
        if self.devices.len() != 1 {
            return Err("device UUID requires a single-device engine".into());
        }
        let uuid = self.devices[0].ctx.uuid()?;
        let hex = uuid
            .bytes
            .iter()
            .map(|&byte| format!("{:02x}", byte as u8))
            .collect::<String>();
        Ok(format!(
            "GPU-{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        ))
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn clear_worker_resources() {
        WORKER_BUFFERS.with(|b| {
            *b.borrow_mut() = None;
        });
        ASSIGNED_DEVICE.with(|a| {
            *a.borrow_mut() = None;
        });
        DEVICE_LOST.with(|lost| {
            *lost.borrow_mut() = None;
        });
    }

    /// Hashes `count` consecutive nonces on the first device with the mining
    /// kernel's arithmetic. This is a kernel self-test under the engine's
    /// probabilistic contract, not a verifier; use `pow_core::hash_from_nonce`
    /// for exact hashes.
    pub fn hash_nonces(
        &self,
        header: [u8; 32],
        start: U512,
        count: u32,
    ) -> Result<Vec<U512>, Box<dyn std::error::Error>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let device = &self.devices[0];
        let mut buffers = create_buffers(self.engine_id, 0, device, count)?;
        let nonce_be = start.to_big_endian();
        let mid = pow_core::mining_midstate_u32s(header, nonce_be[..32].try_into().unwrap());
        let start_limbs = pow_core::u512_to_le_u32s(start);
        buffers.stream.memcpy_htod(&mid, &mut buffers.midstate)?;
        buffers
            .stream
            .memcpy_htod(&start_limbs, &mut buffers.start_nonce)?;

        let num_blocks = count.div_ceil(THREADS_PER_BLOCK).max(1);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.hash);
        builder.arg(&mut buffers.hashes);
        builder.arg(&buffers.midstate);
        builder.arg(&buffers.start_nonce);
        builder.arg(&count);
        unsafe {
            builder.launch(cfg)?;
        }
        buffers.stream.synchronize()?;
        let raw = buffers.stream.clone_dtoh(&buffers.hashes)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let mut limbs = [0u32; 16];
            limbs.copy_from_slice(&raw[i * 16..(i + 1) * 16]);
            out.push(pow_core::u512_from_le_u32s(limbs));
        }
        Ok(out)
    }
}

fn nvrtc_supported_archs() -> Result<Vec<i32>, String> {
    let mut count = 0i32;
    unsafe { nvrtc_sys::nvrtcGetNumSupportedArchs(&mut count) }
        .result()
        .map_err(|e| format!("nvrtcGetNumSupportedArchs failed: {e:?}"))?;
    let mut archs = vec![0i32; count.max(0) as usize];
    unsafe { nvrtc_sys::nvrtcGetSupportedArchs(archs.as_mut_ptr()) }
        .result()
        .map_err(|e| format!("nvrtcGetSupportedArchs failed: {e:?}"))?;
    Ok(archs)
}

/// Compiles PTX for the newest virtual architecture NVRTC knows that the device
/// can run. The driver JIT then emits native code; on a 4090 that measured
/// equal to loading NVRTC's own cubin, so the PTX route is kept.
fn compile_kernel(major: i32, minor: i32) -> Result<Ptx, Box<dyn std::error::Error>> {
    let (mut nvrtc_major, mut nvrtc_minor) = (0, 0);
    unsafe { nvrtc_sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) }
        .result()
        .map_err(|e| format!("nvrtcVersion failed: {e:?}"))?;
    log::info!(target: "cuda_engine", "NVRTC compiler version {nvrtc_major}.{nvrtc_minor}");
    let device_arch = major * 10 + minor;
    let target = nvrtc_supported_archs()?
        .into_iter()
        .filter(|&arch| arch <= device_arch)
        .max()
        .ok_or_else(|| format!("NVRTC supports no architecture at or below sm_{device_arch}"))?;
    if target != device_arch {
        log::warn!(
            target: "cuda_engine",
            "NVRTC does not know sm_{device_arch}; compiling PTX for compute_{target} instead"
        );
    }
    let opts = CompileOptions {
        options: vec![format!("--gpu-architecture=compute_{target}")],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| {
        format!("NVRTC failed to compile the CUDA mining kernel for compute_{target}: {e}")
    })?;
    log::info!(
        target: "cuda_engine",
        "Compiled CUDA mining kernel PTX for compute_{target} (device sm_{device_arch}); the driver JIT emits native code"
    );
    Ok(ptx)
}

fn silent_catch<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    out
}

fn create_buffers(
    engine_id: usize,
    device_index: usize,
    device: &CudaDevice,
    hash_capacity: u32,
) -> Result<WorkerBuffers, Box<dyn std::error::Error>> {
    let stream = device.ctx.default_stream();
    let mine = device.module.load_function("mining_main")?;
    let hash = device.module.load_function("hash_nonces")?;
    Ok(WorkerBuffers {
        engine_id,
        device_index,
        results: stream.alloc_zeros::<u32>(RESULTS_U32S)?,
        midstate: stream.alloc_zeros::<u32>(24)?,
        start_nonce: stream.alloc_zeros::<u32>(16)?,
        hashes: stream.alloc_zeros::<u32>(hash_capacity.max(1) as usize * 16)?,
        mine,
        hash,
        stream,
        busy: Duration::ZERO,
        busy_since: Instant::now(),
    })
}

fn take_gpu_duty_cycle(buffers: &mut WorkerBuffers) -> Option<f64> {
    let wall = buffers.busy_since.elapsed();
    let duty = (wall >= Duration::from_secs(1))
        .then(|| 100.0 * buffers.busy.as_secs_f64() / wall.as_secs_f64());
    buffers.busy = Duration::ZERO;
    buffers.busy_since = Instant::now();
    duty
}

fn worker_buffers(
    engine: &CudaEngine,
    device_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    WORKER_BUFFERS.with(|cell| {
        let mut slot = cell.borrow_mut();
        let reuse = matches!(
            &*slot,
            Some(b) if b.engine_id == engine.engine_id && b.device_index == device_index
        );
        if !reuse {
            *slot = Some(create_buffers(
                engine.engine_id,
                device_index,
                &engine.devices[device_index],
                1,
            )?);
        }
        Ok(())
    })
}

enum BatchResult {
    Found {
        candidate: Candidate,
        hash_count: u64,
    },
    NotFound {
        hash_count: u64,
    },
    Overflow {
        hash_count: u64,
    },
    DeviceLost,
}

/// Tracks logical progress separately from physical hashes spent on replays.
struct BatchCursor {
    start: U512,
    end: U512,
    configured_cap: u32,
    cap: u32,
}

impl BatchCursor {
    fn new(range: Range, cap: u32) -> Self {
        Self {
            start: range.start,
            end: range.end,
            configured_cap: cap,
            cap,
        }
    }

    fn batch_size(&self) -> u32 {
        let remaining = self
            .end
            .saturating_sub(self.start)
            .saturating_add(U512::one());
        remaining
            .min(nonces_until_low64_carry(self.start))
            .min(U512::from(self.cap))
            .low_u32()
    }

    fn replay_overflow(&mut self, batch_size: u32) {
        // An overflowing launch has more than MAX_HITS nonces, so halving
        // strictly shrinks it. A batch of at most MAX_HITS cannot overflow.
        self.cap = (batch_size / 2).max(1);
    }

    fn advance(&mut self, batch_size: u32) -> bool {
        // Compare the last evaluated nonce instead of saturating the next
        // start: saturation would replay U512::MAX forever.
        if U512::from(batch_size - 1) >= self.end - self.start {
            return false;
        }
        self.start += U512::from(batch_size);
        self.cap = self.configured_cap;
        true
    }
}

fn nonces_until_low64_carry(nonce: U512) -> U512 {
    (U512::one() << 64).saturating_sub(U512::from(nonce.low_u64()))
}

fn run_single_batch(
    buffers: &mut WorkerBuffers,
    ctx: &JobContext,
    batch_start: U512,
    batch_size: u32,
) -> BatchResult {
    let num_blocks = batch_size.div_ceil(THREADS_PER_BLOCK).clamp(1, MAX_BLOCKS);
    let total_threads = num_blocks * THREADS_PER_BLOCK;
    let nonces_per_thread = batch_size.div_ceil(total_threads).max(1);
    let dispatch = [total_threads, nonces_per_thread, batch_size];
    let nonce_be = batch_start.to_big_endian();
    let start_limbs = pow_core::u512_to_le_u32s(batch_start);
    let prestate = pow_core::mining_prestate_low64_u32s(ctx.header, nonce_be);
    let target = pow_core::u512_to_le_u32s(ctx.target);
    let params = MiningParams {
        prestate,
        start_nonce: start_limbs,
        difficulty_target: target,
        dispatch_config: dispatch,
    };

    let launch_start = Instant::now();
    if let Err(e) = (|| {
        buffers.stream.memset_zeros(&mut buffers.results)?;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.mine);
        builder.arg(&mut buffers.results);
        builder.arg(&params);
        unsafe {
            builder.launch(cfg)?;
        }
        buffers.stream.synchronize()?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })() {
        log::error!(target: "cuda_engine", "CUDA batch failed: {e}");
        return BatchResult::DeviceLost;
    }
    buffers.busy += launch_start.elapsed();

    let result_u32s = match buffers.stream.clone_dtoh(&buffers.results) {
        Ok(v) => v,
        Err(e) => {
            log::error!(target: "cuda_engine", "CUDA result copy failed: {e}");
            return BatchResult::DeviceLost;
        }
    };
    // cudarc 0.19.9 submits an asynchronous copy into ordinary host memory.
    // The pre-copy fence above completes the kernel, not this later copy.
    // Fence readback explicitly before inspecting the host result buffer.
    if let Err(e) = buffers.stream.synchronize() {
        log::error!(target: "cuda_engine", "CUDA result copy completion failed: {e}");
        return BatchResult::DeviceLost;
    }

    let dispatched = (total_threads as u64 * nonces_per_thread as u64).min(batch_size as u64);
    let hits = result_u32s[0] as usize;
    if hits > MAX_HITS {
        log::warn!(
            target: "cuda_engine",
            "CUDA launch produced {hits} candidates, more than {MAX_HITS} fit; replaying a smaller batch"
        );
        return BatchResult::Overflow {
            hash_count: dispatched,
        };
    }
    let mut best: Option<(U512, U512)> = None;
    for &index in &result_u32s[1..1 + hits] {
        let logical_index = index as u64;
        if logical_index >= dispatched {
            log::error!(
                target: "cuda_engine",
                "CUDA returned out-of-range candidate index {logical_index} for {dispatched} dispatched nonces"
            );
            return BatchResult::DeviceLost;
        }
        let nonce = batch_start + U512::from(logical_index);
        let hash = pow_core::hash_from_nonce(ctx, nonce);
        if hash >= ctx.target {
            log::warn!(
                target: "cuda_engine",
                "CUDA candidate {} rejected by CPU verification (hash {} >= target {})",
                format_u512(nonce),
                format_u512(hash),
                format_u512(ctx.target)
            );
            continue;
        }
        if best.is_none_or(|(n, _)| nonce < n) {
            best = Some((nonce, hash));
        }
    }
    match best {
        Some((nonce, hash)) => BatchResult::Found {
            candidate: Candidate {
                nonce,
                work: nonce.to_big_endian(),
                hash,
            },
            hash_count: dispatched,
        },
        None => BatchResult::NotFound {
            hash_count: dispatched,
        },
    }
}

impl MinerEngine for CudaEngine {
    fn name(&self) -> &'static str {
        "gpu-cuda"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn search_range(
        &self,
        ctx: &JobContext,
        range: Range,
        cancel: &dyn CancelCheck,
    ) -> EngineStatus {
        if self.devices.is_empty() {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if DEVICE_LOST.with(|lost| *lost.borrow() == Some(self.engine_id)) {
            return EngineStatus::DeviceLost { hash_count: 0 };
        }
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if cancel.is_cancelled() {
            return EngineStatus::Cancelled { hash_count: 0 };
        }

        let device_index = ASSIGNED_DEVICE.with(|assigned| {
            let mut assigned_ref = assigned.borrow_mut();
            match *assigned_ref {
                Some((engine_id, index)) if engine_id == self.engine_id => index,
                _ => {
                    let index = if self.devices.len() == 1 {
                        0
                    } else {
                        self.device_counter.fetch_add(1, Ordering::SeqCst) % self.devices.len()
                    };
                    *assigned_ref = Some((self.engine_id, index));
                    log::info!(
                        target: "cuda_engine",
                        "Worker thread assigned to CUDA device {} ({})",
                        index,
                        self.devices[index].name
                    );
                    index
                }
            }
        });

        if let Err(e) = worker_buffers(self, device_index) {
            log::error!(target: "cuda_engine", "CUDA buffer setup failed: {e}");
            DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
            return EngineStatus::DeviceLost { hash_count: 0 };
        }

        let search_start = Instant::now();
        let mut total_hashes: u64 = 0;
        let mut cursor = BatchCursor::new(range.clone(), self.batch_size);
        let mut batch_num = 0u64;

        let duty = WORKER_BUFFERS.with(|cell| {
            take_gpu_duty_cycle(
                cell.borrow_mut()
                    .as_mut()
                    .expect("CUDA buffers initialized"),
            )
        });
        log::info!(
            target: "cuda_engine",
            "CUDA device {} search started: range {}..{}, batch size: {} nonces{}",
            device_index,
            format_u512(range.start),
            format_u512(range.end),
            self.batch_size,
            duty.map_or(String::new(), |d| format!(", GPU busy {d:.1}% since previous search"))
        );

        loop {
            if cancel.is_cancelled() {
                return EngineStatus::Cancelled {
                    hash_count: total_hashes,
                };
            }

            let this_batch_size = cursor.batch_size();

            let batch_result = WORKER_BUFFERS.with(|cell| {
                let mut slot = cell.borrow_mut();
                let buffers = slot.as_mut().expect("CUDA buffers initialized");
                run_single_batch(buffers, ctx, cursor.start, this_batch_size)
            });

            match batch_result {
                BatchResult::Found {
                    candidate,
                    hash_count,
                } => {
                    total_hashes = total_hashes.saturating_add(hash_count);
                    return EngineStatus::Found {
                        candidate,
                        hash_count: total_hashes,
                        origin: FoundOrigin::Cuda,
                    };
                }
                BatchResult::NotFound { hash_count } => {
                    total_hashes = total_hashes.saturating_add(hash_count);
                }
                BatchResult::Overflow { hash_count } => {
                    total_hashes = total_hashes.saturating_add(hash_count);
                    cursor.replay_overflow(this_batch_size);
                    continue;
                }
                BatchResult::DeviceLost => {
                    DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
                    WORKER_BUFFERS.with(|res| *res.borrow_mut() = None);
                    return EngineStatus::DeviceLost {
                        hash_count: total_hashes,
                    };
                }
            }

            if !cursor.advance(this_batch_size) {
                break;
            }
            batch_num += 1;

            if self.throttle_ms > 0 {
                let sleep_interval =
                    std::time::Duration::from_millis((self.throttle_ms / 10).max(1));
                let mut remaining = std::time::Duration::from_millis(self.throttle_ms);
                while remaining > std::time::Duration::ZERO {
                    if cancel.is_cancelled() {
                        return EngineStatus::Cancelled {
                            hash_count: total_hashes,
                        };
                    }
                    let sleep_time = remaining.min(sleep_interval);
                    std::thread::sleep(sleep_time);
                    remaining = remaining.saturating_sub(sleep_time);
                }
            }

            if batch_num.is_multiple_of(10) {
                let elapsed = search_start.elapsed();
                let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
                log::debug!(
                    target: "cuda_engine",
                    "CUDA device {} batch {} complete: {} hashes so far ({:.2}s, {})",
                    device_index,
                    batch_num,
                    total_hashes,
                    elapsed.as_secs_f64(),
                    format_hashrate(hash_rate)
                );
            }
        }

        EngineStatus::Exhausted {
            hash_count: total_hashes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::nvrtc::compile_ptx;
    use engine_cpu::AtomicBoolCancelCheck;
    use std::sync::atomic::AtomicBool;

    fn decode32(s: &str) -> [u8; 32] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn decode64(s: &str) -> [u8; 64] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn engine_or_skip() -> Option<CudaEngine> {
        engine_or_skip_with(1024)
    }

    fn engine_or_skip_with(batch_size: u32) -> Option<CudaEngine> {
        match CudaEngine::try_new(batch_size, 0) {
            Ok(e) => Some(e),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not available") || msg.contains("No CUDA devices") {
                    eprintln!("skipping CUDA test: {e}");
                    None
                } else {
                    panic!("CUDA engine init failed: {e}");
                }
            }
        }
    }

    #[test]
    fn low64_batches_stop_before_carry() {
        assert_eq!(nonces_until_low64_carry(U512::from(u64::MAX)), U512::one());
        assert_eq!(
            nonces_until_low64_carry(U512::from(u64::MAX - 7)),
            U512::from(8u64)
        );
        assert_eq!(
            nonces_until_low64_carry((U512::one() << 192) + U512::from(1u64)),
            (U512::one() << 64) - U512::one()
        );
    }

    #[test]
    fn overflow_replays_same_start_and_restores_cap_after_progress() {
        let mut cursor = BatchCursor::new(
            Range {
                start: U512::from(100u64),
                end: U512::from(200u64),
            },
            33,
        );
        let mut physical_hashes = 0;
        for expected in [33, 16, 8] {
            assert_eq!(cursor.batch_size(), expected);
            assert_eq!(cursor.start, U512::from(100u64));
            physical_hashes += expected;
            if expected > MAX_HITS as u32 {
                cursor.replay_overflow(expected);
            }
        }
        assert_eq!(physical_hashes, 57);
        assert!(cursor.advance(8));
        assert_eq!(cursor.start, U512::from(108u64));
        assert_eq!(cursor.batch_size(), 33);
    }

    #[test]
    fn dense_four_million_batch_counts_all_physical_overflow_replays() {
        let start = U512::from(0xfeed_face_0000_0000u64);
        let mut cursor = BatchCursor::new(
            Range {
                start,
                end: start + U512::from(3_999_999u32),
            },
            4_000_000,
        );
        let expected_launches = [
            4_000_000, 2_000_000, 1_000_000, 500_000, 250_000, 125_000, 62_500, 31_250, 15_625,
            7_812, 3_906, 1_953, 976, 488, 244, 122, 61, 30, 15, 7,
        ];
        assert_eq!(MAX_HITS, 8);
        let mut physical_hashes = 0u64;
        for expected in expected_launches {
            let batch_size = cursor.batch_size();
            assert_eq!(batch_size, expected);
            assert_eq!(cursor.start, start, "overflow must not advance the nonce");
            physical_hashes += u64::from(batch_size);
            if batch_size > MAX_HITS as u32 {
                cursor.replay_overflow(batch_size);
            }
        }
        // Every nonce is a candidate: 19 overflowing launches followed by
        // one fitting launch of 7 nonces. All physical work is counted.
        assert_eq!(cursor.batch_size(), 7);
        assert_eq!(physical_hashes, 7_999_989);
    }

    #[test]
    fn batch_cursor_reduction_terminates_and_respects_range_and_carry() {
        let mut cursor = BatchCursor::new(
            Range {
                start: U512::from(u64::MAX - 10),
                end: U512::from(u64::MAX) + U512::from(100u64),
            },
            u32::MAX,
        );
        assert_eq!(cursor.batch_size(), 11);
        cursor.replay_overflow(11);
        assert_eq!(cursor.batch_size(), 5);
        assert!(cursor.advance(5));
        assert_eq!(cursor.batch_size(), 6);
        assert!(cursor.advance(6));
        assert_eq!(cursor.start, U512::from(u64::MAX) + U512::one());
        assert_eq!(cursor.batch_size(), 100);
        assert!(!cursor.advance(100));

        let mut cursor = BatchCursor::new(
            Range {
                start: U512::zero(),
                end: U512::MAX,
            },
            u32::MAX,
        );
        while cursor.batch_size() > MAX_HITS as u32 {
            let previous = cursor.batch_size();
            cursor.replay_overflow(previous);
            assert!(cursor.batch_size() < previous);
        }
        assert!(cursor.batch_size() > 0);
    }

    #[test]
    fn batch_cursor_terminates_at_u512_max_without_wrapping() {
        let mut cursor = BatchCursor::new(
            Range {
                start: U512::MAX - U512::from(2u64),
                end: U512::MAX,
            },
            2,
        );
        assert_eq!(cursor.batch_size(), 2);
        assert!(cursor.advance(2));
        assert_eq!(cursor.start, U512::MAX);
        assert_eq!(cursor.batch_size(), 1);
        assert!(!cursor.advance(1));
    }

    #[test]
    fn cuda_dense_target_overflow_returns_lowest_and_all_shares_by_resume() {
        let Some(engine) = engine_or_skip_with(33) else {
            return;
        };
        let ctx = JobContext {
            header: decode32(pow_core::NONCE_HASH_KVS[0].header),
            difficulty: U512::one(),
            target: U512::MAX,
        };
        let cancelled = AtomicBool::new(false);
        let start = U512::from(100u64);
        let end = start + U512::from(32u64);
        // Assert this fixture really fills the device buffer, regardless of
        // atomic claim order, before testing the replay path.
        let mut buffers = create_buffers(engine.engine_id, 0, &engine.devices[0], 1).unwrap();
        assert!(matches!(
            run_single_batch(&mut buffers, &ctx, start, 33),
            BatchResult::Overflow { hash_count: 33 }
        ));
        let expected: Vec<_> = (0..33u64)
            .map(|i| start + U512::from(i))
            .filter(|&nonce| pow_core::hash_from_nonce(&ctx, nonce) < ctx.target)
            .collect();
        let mut resumed_start = start;
        let mut found = Vec::new();
        loop {
            match engine.search_range(
                &ctx,
                Range {
                    start: resumed_start,
                    end,
                },
                &AtomicBoolCancelCheck(&cancelled),
            ) {
                EngineStatus::Found {
                    candidate,
                    hash_count,
                    origin: FoundOrigin::Cuda,
                } => {
                    assert_eq!(
                        candidate.hash,
                        pow_core::hash_from_nonce(&ctx, candidate.nonce)
                    );
                    assert_eq!(candidate.work, candidate.nonce.to_big_endian());
                    if found.is_empty() {
                        assert_eq!(candidate.nonce, expected[0]);
                        assert_eq!(hash_count, 33 + 16 + 8);
                    }
                    found.push(candidate.nonce);
                    if candidate.nonce == end {
                        break;
                    }
                    resumed_start = candidate.nonce + U512::one();
                }
                EngineStatus::Exhausted { .. } => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(found, expected);
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_overflow_replay_observes_cancellation() {
        let Some(engine) = engine_or_skip_with(33) else {
            return;
        };
        struct CancelAfterFirstLaunch(AtomicUsize);
        impl CancelCheck for CancelAfterFirstLaunch {
            fn is_cancelled(&self) -> bool {
                // Initial entry and first loop check pass; the next loop
                // check cancels before the overflowing batch is replayed.
                self.0.fetch_add(1, Ordering::Relaxed) >= 2
            }
        }
        let ctx = JobContext {
            header: decode32(pow_core::NONCE_HASH_KVS[0].header),
            difficulty: U512::one(),
            target: U512::MAX,
        };
        assert!(matches!(
            engine.search_range(
                &ctx,
                Range {
                    start: U512::zero(),
                    end: U512::from(32u64)
                },
                &CancelAfterFirstLaunch(AtomicUsize::new(0)),
            ),
            EngineStatus::Cancelled { hash_count: 33 }
        ));
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_repeated_hit_then_zero_hit_batches_do_not_reuse_stale_results() {
        let Some(engine) = engine_or_skip_with(MAX_HITS as u32) else {
            return;
        };
        let hit_ctx = JobContext {
            header: decode32(pow_core::NONCE_HASH_KVS[0].header),
            difficulty: U512::one(),
            target: U512::MAX,
        };
        let no_hit_ctx = JobContext {
            header: hit_ctx.header,
            difficulty: hit_ctx.difficulty,
            target: U512::zero(),
        };
        let mut buffers = create_buffers(engine.engine_id, 0, &engine.devices[0], 1).unwrap();
        let batch_size = MAX_HITS as u32;
        for round in 0..16u64 {
            let start = U512::from(100 + round * u64::from(batch_size));
            let expected = (0..batch_size)
                .map(|index| start + U512::from(index))
                .find(|&nonce| pow_core::hash_from_nonce(&hit_ctx, nonce) < hit_ctx.target)
                .expect("dense-target fixture must contain a valid nonce");
            match run_single_batch(&mut buffers, &hit_ctx, start, batch_size) {
                BatchResult::Found {
                    candidate,
                    hash_count,
                } => {
                    assert_eq!(candidate.nonce, expected, "hit batch {round}");
                    assert_eq!(candidate.work, expected.to_big_endian());
                    assert_eq!(
                        candidate.hash,
                        pow_core::hash_from_nonce(&hit_ctx, expected)
                    );
                    assert_eq!(hash_count, u64::from(batch_size));
                }
                _ => panic!("hit batch {round} did not return a verified candidate"),
            }
            assert!(
                matches!(
                    run_single_batch(&mut buffers, &no_hit_ctx, start, batch_size),
                    BatchResult::NotFound { hash_count } if hash_count == u64::from(batch_size)
                ),
                "zero-hit batch {round} reused stale results or failed"
            );
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_zero_hits_exhaust_at_u512_max_and_precancel_does_no_work() {
        let Some(engine) = engine_or_skip_with(33) else {
            return;
        };
        let ctx = JobContext {
            header: decode32(pow_core::NONCE_HASH_KVS[0].header),
            difficulty: U512::one(),
            target: U512::zero(),
        };
        let range = Range {
            start: U512::MAX - U512::from(32u64),
            end: U512::MAX,
        };
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            engine.search_range(&ctx, range.clone(), &AtomicBoolCancelCheck(&cancelled)),
            EngineStatus::Cancelled { hash_count: 0 }
        ));
        cancelled.store(false, Ordering::Relaxed);
        assert!(matches!(
            engine.search_range(&ctx, range, &AtomicBoolCancelCheck(&cancelled)),
            EngineStatus::Exhausted { hash_count: 33 }
        ));
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_matches_nonce_hash_golden_vectors() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let header = decode32(v.header);
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let want = U512::from_big_endian(&decode64(v.hash));
            let got = engine
                .hash_nonces(header, nonce, 1)
                .unwrap_or_else(|e| panic!("kv {i}: CUDA hash_nonces failed: {e}"));
            assert_eq!(got.len(), 1, "kv {i}");
            assert_eq!(got[0], want, "kv {i}: CUDA hash != golden");
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_search_matches_golden_target_boundaries() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let cancel = AtomicBool::new(false);
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let hash = U512::from_big_endian(&decode64(v.hash));
            for target in [hash - U512::one(), hash, hash + U512::one()] {
                let ctx = JobContext {
                    header: decode32(v.header),
                    difficulty: U512::one(),
                    target,
                };
                let status = engine.search_range(
                    &ctx,
                    Range {
                        start: nonce,
                        end: nonce,
                    },
                    &AtomicBoolCancelCheck(&cancel),
                );
                match status {
                    EngineStatus::Found { candidate, .. } if hash < target => {
                        assert_eq!(candidate.nonce, nonce, "kv {i}");
                        assert_eq!(candidate.hash, hash, "kv {i}");
                    }
                    EngineStatus::Exhausted { hash_count: 1 } if hash >= target => {}
                    other => panic!("kv {i}, target {target}: unexpected {other:?}"),
                }
            }
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_hash_batches_match_cpu_across_nonce_carries() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[4].header);
        let ctx = engine.prepare_context(header, U512::one());
        let high = U512::from(0xdead_beef_cafeu64) << 256;
        for bits in [32, 64, 128, 224] {
            let start = high + (U512::one() << bits) - U512::from(128u64);
            let hashes = engine.hash_nonces(header, start, 257).unwrap();
            for (i, hash) in hashes.into_iter().enumerate() {
                let nonce = start + U512::from(i);
                assert_eq!(
                    hash,
                    pow_core::hash_from_nonce(&ctx, nonce),
                    "nonce {nonce}"
                );
            }
        }
        CudaEngine::clear_worker_resources();
    }

    const GOLDILOCKS_P: u128 = 0xffff_ffff_0000_0001;
    const EPS: u64 = 0xffff_ffff;

    /// Bit-exact model of the kernel's `reduce128` and its deviation from
    /// `value mod p`, so the GPU can be checked exactly and the deviation
    /// bounded on the host.
    fn reduce128_model(value: u128) -> (u64, u128) {
        let (r0, r1, r2, r3) = (
            value as u32,
            (value >> 32) as u32,
            (value >> 64) as u32,
            (value >> 96) as u32,
        );
        let low = ((r1 as u64) << 32) | r0 as u64;
        let (folded, carry) = low.overflowing_add(r2 as u64 * EPS);
        let (subtrahend, subtrahend_wrapped) = r3.overflowing_add(carry as u32);
        let (hi, hi_wrapped) = ((folded >> 32) as u32).overflowing_add(carry as u32);
        let shifted = ((hi as u64) << 32) | (folded as u32 as u64);
        let (result, borrowed) = shifted.overflowing_sub(subtrahend as u64);
        let mut error = 0u128;
        if subtrahend_wrapped {
            error += 1 << 32;
        }
        if hi_wrapped {
            error += GOLDILOCKS_P - EPS as u128;
        }
        if borrowed {
            error += EPS as u128;
        }
        (result, error % GOLDILOCKS_P)
    }

    /// Bit-exact model of the kernel's `gf64_add` and its deviation from `a + b mod p`.
    fn gf64_add_model(a: u64, b: u64) -> (u64, u128) {
        let (sum, carry) = a.overflowing_add(b);
        if !carry {
            return (sum, 0);
        }
        let (folded, wrapped) = sum.overflowing_add(EPS);
        (
            folded,
            if wrapped {
                GOLDILOCKS_P - EPS as u128
            } else {
                0
            },
        )
    }

    /// Edge cases built to hit every carry path, followed by 4096 random values.
    fn reduction_inputs() -> (Vec<u128>, Vec<u128>) {
        let edges = [
            0,
            1,
            0xffff_fffe,
            0xffff_ffff,
            0x1_0000_0000,
            GOLDILOCKS_P as u64 - 1,
            GOLDILOCKS_P as u64,
            GOLDILOCKS_P as u64 + 1,
            u64::MAX,
        ];
        let mut edge_values = Vec::new();
        for high in edges {
            for low in edges {
                edge_values.push(((high as u128) << 64) | low as u128);
            }
        }
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut random = Vec::new();
        for _ in 0..4096 {
            let mut value = 0u128;
            for _ in 0..2 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                value = (value << 64) | seed as u128;
            }
            random.push(value);
        }
        (edge_values, random)
    }

    #[test]
    fn reduction_models_deviate_only_on_carry_wraps() {
        let (edge_values, random) = reduction_inputs();
        for value in edge_values.iter().chain(&random) {
            let (result, error) = reduce128_model(*value);
            assert_eq!(
                result as u128 % GOLDILOCKS_P,
                (value % GOLDILOCKS_P + error) % GOLDILOCKS_P,
                "reduce128 model invariant broken for {value:032x}"
            );
        }
        assert_eq!(reduce128_model(1 << 96).1, EPS as u128, "2^96 borrows");
        let slips = random
            .iter()
            .filter(|v| reduce128_model(**v).1 != 0)
            .count();
        assert_eq!(
            slips,
            0,
            "slipped reductions among {} random inputs",
            random.len()
        );

        let mut random_slips = 0;
        for pair in random.windows(2) {
            let (a, b) = (pair[0] as u64, pair[1] as u64);
            let (result, error) = gf64_add_model(a, b);
            assert_eq!(
                result as u128 % GOLDILOCKS_P,
                (a as u128 + b as u128 + error) % GOLDILOCKS_P,
                "gf64_add model invariant broken for {a:016x} + {b:016x}"
            );
            random_slips += (error != 0) as usize;
        }
        assert_eq!(
            gf64_add_model(u64::MAX, u64::MAX).1,
            GOLDILOCKS_P - EPS as u128
        );
        assert_eq!(random_slips, 0, "slipped additions among random inputs");
    }

    #[test]
    fn reduction_slips_are_absent_in_millions_of_random_products() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let slips = (0..1 << 22)
            .filter(|_| reduce128_model(next() as u128 * next() as u128).1 != 0)
            .count();
        assert_eq!(slips, 0, "reducer slipped on random 64x64-bit products");
    }

    #[test]
    fn cuda_field_arithmetic_matches_models_bit_exactly() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let (edge_values, random) = reduction_inputs();
        let values: Vec<u128> = edge_values.into_iter().chain(random).collect();
        let input: Vec<u32> = values
            .iter()
            .flat_map(|v| (0..4).map(move |i| (v >> (32 * i)) as u32))
            .collect();
        let source = format!(
            "{KERNEL_SRC}\n\
             extern \"C\" __global__ void arith_test(const u32 *input, u64 *reduced, u64 *added, u32 count) {{\n\
                 u32 i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 if (i >= count) return;\n\
                 reduced[i] = reduce128(input[4*i], input[4*i+1], input[4*i+2], input[4*i+3]);\n\
                 u64 a = ((u64)input[4*i+1] << 32) | input[4*i];\n\
                 u64 b = ((u64)input[4*i+3] << 32) | input[4*i+2];\n\
                 added[i] = gf64_add(a, b);\n\
             }}"
        );
        let device = &engine.devices[0];
        let module = device
            .ctx
            .load_module(compile_ptx(source).unwrap())
            .unwrap();
        let function = module.load_function("arith_test").unwrap();
        let stream = device.ctx.default_stream();
        let input = stream.clone_htod(&input).unwrap();
        let mut reduced = stream.alloc_zeros::<u64>(values.len()).unwrap();
        let mut added = stream.alloc_zeros::<u64>(values.len()).unwrap();
        let count = values.len() as u32;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&input)
            .arg(&mut reduced)
            .arg(&mut added)
            .arg(&count);
        unsafe {
            launch.launch(LaunchConfig::for_num_elems(count)).unwrap();
        }
        stream.synchronize().unwrap();
        let reduced = stream.clone_dtoh(&reduced).unwrap();
        let added = stream.clone_dtoh(&added).unwrap();
        for (i, value) in values.iter().enumerate() {
            assert_eq!(
                reduced[i],
                reduce128_model(*value).0,
                "reduce128 of {value:032x}"
            );
            let (a, b) = (*value as u64, (*value >> 64) as u64);
            assert_eq!(
                added[i],
                gf64_add_model(a, b).0,
                "gf64_add of {a:016x} + {b:016x}"
            );
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_found_batch_counts_every_dispatched_nonce() {
        let batch_size = 4_000_000u32;
        let Some(engine) = engine_or_skip_with(batch_size) else {
            return;
        };
        // Difficulty 1 makes every nonce a candidate. Each launch evaluates
        // its whole batch; overflowing launches replay the same start at half
        // size until 7 candidates fit. Count all 20 launches (7,999,989 hashes)
        // and return the lowest CPU-valid nonce, rather than an arbitrary claim.
        let header = decode32(pow_core::NONCE_HASH_KVS[1].header);
        let ctx = engine.prepare_context(header, U512::one());
        let start = U512::from(0xfeed_face_0000_0000u64);
        let end = start + U512::from(batch_size - 1);
        assert!(pow_core::hash_from_nonce(&ctx, start) < ctx.target);
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, Range { start, end }, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Found {
                candidate,
                hash_count,
                ..
            } => {
                assert_eq!(candidate.nonce, start);
                assert_eq!(
                    pow_core::hash_from_nonce(&ctx, candidate.nonce),
                    candidate.hash
                );
                assert!(candidate.hash < ctx.target);
                assert_eq!(hash_count, 7_999_989);
            }
            other => panic!("expected Found, got {other:?}"),
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_search_rejects_prefix_equal_candidate_and_returns_lowest_valid() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        // Target equal to the golden hash: the golden nonce at index 0 is a
        // prefix-equal candidate that CPU verification rejects. The range holds
        // MAX_HITS nonces so every candidate is recorded, and the engine must
        // return exactly the lowest nonce the CPU finds valid, or Exhausted if
        // there is none.
        let v = &pow_core::NONCE_HASH_KVS[4];
        let start = U512::from_big_endian(&decode64(v.nonce));
        let target = U512::from_big_endian(&decode64(v.hash));
        let ctx = JobContext {
            header: decode32(v.header),
            difficulty: U512::one(),
            target,
        };
        let count = MAX_HITS as u64;
        let end = start + U512::from(count - 1);
        let lowest_valid = (0..count)
            .map(|i| start + U512::from(i))
            .find(|&nonce| pow_core::hash_from_nonce(&ctx, nonce) < target);
        assert_ne!(
            lowest_valid,
            Some(start),
            "the golden nonce itself must not qualify"
        );
        let cancel = AtomicBool::new(false);
        match (
            engine.search_range(&ctx, Range { start, end }, &AtomicBoolCancelCheck(&cancel)),
            lowest_valid,
        ) {
            (
                EngineStatus::Found {
                    candidate,
                    hash_count,
                    ..
                },
                Some(expected),
            ) => {
                assert_eq!(candidate.nonce, expected);
                assert_eq!(candidate.hash, pow_core::hash_from_nonce(&ctx, expected));
                assert_eq!(hash_count, count);
            }
            (EngineStatus::Exhausted { hash_count }, None) => assert_eq!(hash_count, count),
            (other, expected) => panic!("unexpected {other:?} for expected {expected:?}"),
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_search_finds_cpu_verified_solution() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[2].header);
        let difficulty = U512::from(1u64);
        let ctx = engine.prepare_context(header, difficulty);
        let start = U512::from(0x1234_5678_90ab_cdefu64);
        let range = Range {
            start,
            end: start + U512::from(16u64),
        };
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, range, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Found { candidate, .. } => {
                let cpu = pow_core::hash_from_nonce(&ctx, candidate.nonce);
                assert_eq!(cpu, candidate.hash);
                assert!(cpu < ctx.target);
            }
            other => panic!("expected Found, got {other:?}"),
        }
        CudaEngine::clear_worker_resources();
    }
}

#[cfg(test)]
mod device_selection_tests;
