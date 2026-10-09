use super::select_ordinals;

#[test]
fn explicit_visible_ordinals_preserve_order_and_do_not_select_other_devices() {
    assert_eq!(select_ordinals(4, Some(&[3, 1])).unwrap(), vec![3, 1]);
    assert_eq!(select_ordinals(4, Some(&[0])).unwrap(), vec![0]);
    assert_eq!(select_ordinals(3, None).unwrap(), vec![0, 1, 2]);
}

#[test]
fn invalid_explicit_selection_fails_closed() {
    for ordinals in [&[][..], &[1, 1][..], &[4][..], &[usize::MAX][..]] {
        assert!(select_ordinals(4, Some(ordinals)).is_err());
    }
    assert!(select_ordinals(0, None).is_err());
}
