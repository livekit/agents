use super::*;

fn named(key: &str) -> BindingKey {
    BindingKey::Named(key.into())
}

#[test]
fn closing_a_binding_keeps_its_key_until_cleanup_settles() {
    let mut owner = RoomBindings::default();
    let first = BindingId(1);
    let replacement = BindingId(2);
    owner.declare(first, named("voice")).unwrap();
    assert_eq!(
        owner.declare(replacement, named("voice")),
        Err(BindingRefused::KeyInUse(named("voice")))
    );
    owner.begin_start(first).unwrap();
    assert!(owner.started(first));
    assert!(!owner.retired(first));
    assert_eq!(owner.close_binding(first), Some(RoomBindingEffect::Close(first)));
    assert_eq!(
        owner.declare(replacement, named("voice")),
        Err(BindingRefused::KeyInUse(named("voice")))
    );
    assert!(owner.retired(first));
    owner.declare(replacement, named("voice")).unwrap();
}

#[test]
fn late_completion_and_cleanup_cannot_touch_a_reused_key() {
    let mut owner = RoomBindings::default();
    let old = BindingId(1);
    let new = BindingId(2);
    owner.declare(old, named("translation")).unwrap();
    owner.begin_start(old).unwrap();
    owner.close_binding(old);
    assert!(!owner.started(old));
    assert!(owner.retired(old));
    owner.declare(new, named("translation")).unwrap();
    owner.begin_start(new).unwrap();
    assert!(owner.started(new));
    assert!(!owner.started(old));
    assert!(!owner.retired(old));
    assert_eq!(owner.close_binding(old), None);
    assert_eq!(owner.phase(new), Some(BindingPhase::Active));
}

#[test]
fn two_consumers_cannot_activate_the_same_binding() {
    let mut owner = RoomBindings::default();
    let id = BindingId(1);
    owner.declare(id, named("captions")).unwrap();
    owner.begin_start(id).unwrap();
    assert_eq!(owner.begin_start(id), Err(BindingRefused::Unavailable));
    assert!(owner.started(id));
    assert!(!owner.started(id));
    assert_eq!(owner.begin_start(id), Err(BindingRefused::Unavailable));
}

#[test]
fn retired_incarnation_ids_cannot_be_recycled() {
    let mut owner = RoomBindings::default();
    let id = BindingId(1);
    owner.declare(id, named("voice")).unwrap();
    owner.close_binding(id);
    assert!(owner.retired(id));
    assert_eq!(owner.declare(id, named("replacement")), Err(BindingRefused::Unavailable));
    assert_eq!(owner.declare(BindingId(0), named("older")), Err(BindingRefused::Unavailable));
}

#[test]
fn convenience_bindings_cannot_collide_with_application_names() {
    let mut owner = RoomBindings::default();
    owner.declare(BindingId(1), BindingKey::Automatic(BindingId(1))).unwrap();
    owner.declare(BindingId(2), named("1")).unwrap();
    owner.declare(BindingId(3), named("Automatic(BindingId(1))")).unwrap();
    assert_eq!(owner.phase(BindingId(1)), Some(BindingPhase::Declared));
}

#[test]
fn owner_close_stops_admission_and_waits_for_every_binding() {
    let mut owner = RoomBindings::default();
    let declared = BindingId(1);
    let starting = BindingId(2);
    let active = BindingId(3);
    let closing = BindingId(4);
    for id in [declared, starting, active, closing] {
        owner.declare(id, BindingKey::Automatic(id)).unwrap();
    }
    owner.begin_start(starting).unwrap();
    owner.begin_start(active).unwrap();
    owner.started(active);
    owner.close_binding(closing);

    assert_eq!(
        owner.close(),
        vec![
            RoomBindingEffect::Close(declared),
            RoomBindingEffect::Close(starting),
            RoomBindingEffect::Close(active),
        ]
    );
    assert!(!owner.settled());
    assert_eq!(owner.begin_start(declared), Err(BindingRefused::OwnerClosed));
    assert_eq!(owner.declare(BindingId(5), named("new")), Err(BindingRefused::OwnerClosed));
    assert!(!owner.started(starting));
    assert!(owner.close().is_empty());
    for id in [declared, starting, active, closing] {
        assert!(owner.retired(id));
    }
    assert!(owner.settled());
}
