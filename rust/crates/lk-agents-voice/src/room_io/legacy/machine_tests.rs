use super::*;

#[test]
fn successful_activation_order_selects_primary_and_duplicate_receipts_do_not_steal_it() {
    let mut bridge = LegacyPrimary::default();
    let first = BindingId(1);
    let second = BindingId(2);
    bridge.started(second);
    // Declaration order is irrelevant: the first declared binding finished startup last.
    bridge.started(first);
    let projection = bridge.current().unwrap();
    assert_eq!(projection.binding, first);
    assert_eq!(bridge.started(second), None);
    assert_eq!(bridge.current(), Some(projection));
}

#[test]
fn switching_away_and_back_does_not_authorize_old_events() {
    let mut bridge = LegacyPrimary::default();
    let first = BindingId(1);
    let second = BindingId(2);
    bridge.started(first);
    let admitted = bridge.current().unwrap();
    bridge.started(second);
    assert!(!bridge.accepts_event(admitted));
    bridge.pin(first).unwrap();
    let current = bridge.current().unwrap();
    assert_eq!(admitted.binding, current.binding);
    assert_ne!(admitted.generation, current.generation);
    assert!(!bridge.accepts_event(admitted));
    assert!(bridge.accepts_event(current));
    // A request admitted before cutover still holds its original target and correlation.
    assert_eq!(admitted.binding, first);
}

#[test]
fn an_explicit_pin_survives_other_starts_until_automatic_selection_is_requested() {
    let mut bridge = LegacyPrimary::default();
    let pinned = BindingId(1);
    bridge.started(pinned);
    bridge.pin(pinned).unwrap();
    let projection = bridge.current().unwrap();
    bridge.started(BindingId(2));
    bridge.started(BindingId(3));
    assert_eq!(bridge.current(), Some(projection));
    bridge.automatic();
    assert_eq!(bridge.pinned(), None);
    assert_eq!(bridge.current().unwrap().binding, BindingId(3));
    bridge.started(BindingId(4));
    assert_eq!(bridge.current().unwrap().binding, BindingId(4));
}

#[test]
fn primary_loss_does_not_promote_another_active_binding() {
    let mut bridge = LegacyPrimary::default();
    bridge.started(BindingId(1));
    bridge.started(BindingId(2));
    let retired = bridge.current().unwrap();
    assert_eq!(
        bridge.closing(BindingId(2)),
        Some(LegacyPrimaryEffect::Cutover { previous: Some(retired), current: None })
    );
    assert_eq!(bridge.current(), None);
    assert!(!bridge.accepts_event(retired));
    assert_eq!(bridge.closing(BindingId(2)), None);
    bridge.started(BindingId(3));
    assert_eq!(bridge.current().unwrap().binding, BindingId(3));
}

#[test]
fn an_unavailable_pin_does_not_change_the_current_policy_or_target() {
    let mut bridge = LegacyPrimary::default();
    bridge.started(BindingId(1));
    let before = bridge.current();
    assert_eq!(bridge.pin(BindingId(2)), Err(LegacyTargetUnavailable));
    assert_eq!(bridge.current(), before);
    bridge.started(BindingId(3));
    assert_eq!(bridge.current().unwrap().binding, BindingId(3));
}

#[test]
fn closing_a_pinned_target_does_not_give_later_bindings_its_authority() {
    let mut bridge = LegacyPrimary::default();
    let pinned = BindingId(1);
    bridge.started(pinned);
    bridge.pin(pinned).unwrap();
    bridge.closing(pinned);
    bridge.started(BindingId(2));
    assert_eq!(bridge.pinned(), Some(pinned));
    assert_eq!(bridge.current(), None);
    bridge.automatic();
    assert_eq!(bridge.current().unwrap().binding, BindingId(2));
}
