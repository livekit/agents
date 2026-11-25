use crate::{
    authored::{Authored, amd_predicted},
    check::{Check, Contract, Rule, Subject},
    record::{Recording, amd_run::AskKind},
};
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, IvrMenuOption, MenuObserved, Stage};

// Spec §14 invariant 14: a decision model makes one `choice` per classification and no menu.
// Once a run classified with a decision model reaches `machine-ivr`, no menu extraction is
// requested of any model and the host hears of no menu; every choice offers every category.

pub struct DecisionModelExtractsNoMenu;

impl Rule for DecisionModelExtractsNoMenu {
    const CONTRACT: Contract = Contract {
        id: "decision_model_extracts_no_menu",
        guarantee: "A run classified by a decision model asks one choice per turn and never \
                    extracts or reports a menu.",
        examples: &[a_menu_stage_without_extraction],
        counterexamples: &[a_menu_extracted, a_menu_reported],
        probe: Some(a_menu_answers_a_decision_model),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        let choices = view.asks_of(AskKind::Choice);
        if choices.is_empty() {
            return;
        }
        let every: Vec<String> = Category::ALL.iter().map(|c| c.as_str().to_owned()).collect();
        for (nth, _, asked) in &choices {
            check.expect_with(Subject::Resource(crate::record::Kind::AmdAsk, *nth), asked.options == every, || {
                format!("choice {nth} offered {:?}", asked.options)
            });
        }
        for (_, run) in view.entered() {
            let menu_stage = view.predicted().into_iter().find(|p| p.run == run && p.stage == Stage::MachineIvr);
            let Some(entered_menu) = menu_stage else { continue };
            let extracted = view.asks_of(AskKind::Menu).len();
            let reported = view.menus(run);
            check.expect_with(Subject::Session, extracted == 0 && reported == 0, || {
                format!(
                    "run {} reached machine-ivr at {} ms with a decision model, then extracted \
                     {extracted} menus and reported {reported}",
                    run.0, entered_menu.at.at_ms
                )
            });
        }
    }
}

fn ivr() -> Authored {
    let raw = Some(Category::MachineIvr);
    let step = amd_predicted(1, 4, 1, Some(1), raw, Category::MachineIvr, Stage::MachineIvr, AmdReason::Prediction, false);
    Authored::new().amd_entered(1).amd_asked(AskKind::Choice, "{}", "press one").amd_step(step)
}

fn a_menu_stage_without_extraction() -> Recording {
    ivr().done()
}

fn a_menu_extracted() -> Recording {
    ivr().amd_asked(AskKind::Menu, "", r#"{"transcript": "press one"}"#).done()
}

fn a_menu_reported() -> Recording {
    let menu = MenuObserved {
        session_id: "amd".into(),
        turn_id: 1,
        menu: "Main menu".into(),
        options: vec![IvrMenuOption { label: "Sales".into(), dtmf: "1".into(), spoken_response: String::new() }],
        extraction_duration: std::time::Duration::from_millis(400),
    };
    ivr().amd_event(1, AmdEvent::MenuObserved(menu)).done()
}

/// A menu greets a decision model.
fn a_menu_answers_a_decision_model() -> crate::drive::Case {
    use crate::fakes::amd::{AmdSetup, executor_probe::{call, turn}};
    use crate::input::AmdSays;
    let inputs = turn(0, "thanks for calling for billing press one", 0, AmdSays::Ivr, 0);
    call(AmdSetup { decision: true, ..AmdSetup::default() }, inputs)
}
