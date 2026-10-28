use lk_agents_voice::ivr::TfidfLoopDetector;

#[test]
fn repeated_menu_prompts_are_detected_but_a_new_announcement_resets_the_streak() {
    let mut detector = TfidfLoopDetector::default();
    for prompt in
        ["Welcome to the automated phone system", "Press one for sales", "Press two for support"]
    {
        detector.add_chunk(prompt);
        assert!(!detector.check_loop_detection());
    }
    for (prompt, repeats) in [
        ("Press one for sales", false),
        ("Press two for support", false),
        ("Press one for sales", true),
        ("Here is a new announcement never heard before", false),
        ("Press two for support", false),
        ("Press one for sales", false),
        ("Press two for support", true),
    ] {
        detector.add_chunk(prompt);
        assert_eq!(detector.check_loop_detection(), repeats, "{prompt}");
    }
}

#[test]
fn shared_words_in_a_human_conversation_do_not_look_like_a_menu_loop() {
    let mut detector = TfidfLoopDetector::default();
    for prompt in [
        "I need help with my checking account today",
        "Can you confirm the last four digits of the account",
        "There is a thirty five dollar fee that should not be there",
        "Give me a moment while I check those recent transactions",
        "I can reverse that fee and send you a confirmation email",
    ] {
        detector.add_chunk(prompt);
        assert!(!detector.check_loop_detection(), "{prompt}");
    }
}
