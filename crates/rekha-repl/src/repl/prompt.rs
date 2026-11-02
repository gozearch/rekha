//! Dynamic prompt builder.

pub fn build_prompt(collection: &Option<String>) -> reedline::DefaultPrompt {
    let indicator = match collection {
        Some(name) => format!("rekha:{name}> "),
        None => "rekha> ".to_string(),
    };
    reedline::DefaultPrompt::new(
        reedline::DefaultPromptSegment::Basic(indicator),
        reedline::DefaultPromptSegment::Empty,
    )
}
