use super::*;
use crate::room_io::bindings::effects::RoomBindingEffect;
use crate::room_io::legacy::effects::LegacyPrimaryEffect;

fn declare(owner: &mut SharedRoom, id: BindingId, legacy_voice: bool) {
    owner.declare(id, BindingKey::Automatic(id), BindingCapabilities { legacy_voice }).unwrap();
    owner.begin_start(id).unwrap();
}

#[test]
fn non_voice_and_cancelled_starts_cannot_replace_the_legacy_primary() {
    let mut owner = SharedRoom::default();
    let voice = BindingId(1);
    let translation = BindingId(2);
    let cancelled = BindingId(3);
    declare(&mut owner, voice, true);
    owner.started(voice).unwrap();
    let original = owner.legacy_projection().unwrap();
    declare(&mut owner, translation, false);
    assert!(owner.started(translation).unwrap().is_empty());
    assert_eq!(owner.pin_legacy(translation), Err(LegacyTargetUnavailable));
    declare(&mut owner, cancelled, true);
    owner.close_binding(cancelled);
    assert_eq!(owner.started(cancelled), Err(BindingRefused::Unavailable));
    assert_eq!(owner.legacy_projection(), Some(original));
    assert!(owner.accepts_legacy_event(original));
}

#[test]
fn closing_the_primary_withdraws_it_before_resource_cleanup() {
    let mut owner = SharedRoom::default();
    let voice = BindingId(1);
    declare(&mut owner, voice, true);
    owner.started(voice).unwrap();
    let previous = owner.legacy_projection();
    assert_eq!(
        owner.close_binding(voice),
        vec![
            SharedRoomEffect::Legacy(LegacyPrimaryEffect::Cutover { previous, current: None }),
            SharedRoomEffect::Binding(RoomBindingEffect::Close(voice)),
        ]
    );
    assert_eq!(owner.phase(voice), Some(BindingPhase::Closing));
    assert_eq!(owner.legacy_projection(), None);
    assert!(owner.retired(voice));
}

#[test]
fn closing_the_owner_cannot_be_undone_by_a_late_start_or_policy_change() {
    let mut owner = SharedRoom::default();
    let first = BindingId(1);
    let pending = BindingId(2);
    declare(&mut owner, first, true);
    declare(&mut owner, pending, true);
    owner.started(first).unwrap();
    assert_eq!(owner.close().len(), 3);
    assert_eq!(owner.started(pending), Err(BindingRefused::Unavailable));
    assert!(owner.automatic_legacy().is_empty());
    assert_eq!(owner.pin_legacy(first), Err(LegacyTargetUnavailable));
    assert_eq!(owner.legacy_projection(), None);
    assert!(!owner.settled());
    assert!(owner.retired(first));
    assert!(owner.retired(pending));
    assert!(owner.settled());
}

#[test]
fn disabling_legacy_preserves_bindings_without_creating_a_protocol_primary() {
    let mut owner = SharedRoom::new(false);
    let first = BindingId(1);
    let second = BindingId(2);
    declare(&mut owner, first, true);
    declare(&mut owner, second, true);
    assert!(owner.started(first).unwrap().is_empty());
    assert!(owner.started(second).unwrap().is_empty());
    assert_eq!(owner.phase(first), Some(BindingPhase::Active));
    assert_eq!(owner.phase(second), Some(BindingPhase::Active));
    assert_eq!(owner.legacy_projection(), None);
    assert_eq!(owner.pin_legacy(first), Err(LegacyTargetUnavailable));
    assert!(owner.automatic_legacy().is_empty());
}
