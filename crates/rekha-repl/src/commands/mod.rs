//! Command parser for backslash commands.

#[derive(Debug, Clone)]
pub enum Command {
    ListCollections,
    DescribeCollection(String),
    CreateCollection(String),
    UseCollection(String),
    DropCollection(String),
    ToggleVertical,
    SetFormat(String),
    TogglePager,
    ToggleTiming,
    ExportToFile(String),
    ExecuteFile(String),
    Clear,
    Help,
    Quit,
    Unknown(String),
}

pub fn parse_command(input: &str) -> Command {
    let input = input.trim();

    if input == "\\d" || input == "\\dt" {
        return Command::ListCollections;
    }

    if let Some(name) = input
        .strip_prefix("\\d ")
        .or_else(|| input.strip_prefix("\\dt "))
    {
        return Command::DescribeCollection(name.trim().to_string());
    }

    if input == "\\x" {
        return Command::ToggleVertical;
    }

    if let Some(fmt) = input.strip_prefix("\\format ") {
        return Command::SetFormat(fmt.trim().to_string());
    }

    if input == "\\pager" {
        return Command::TogglePager;
    }

    if input == "\\timing" {
        return Command::ToggleTiming;
    }

    if let Some(file) = input.strip_prefix("\\o ") {
        return Command::ExportToFile(file.trim().to_string());
    }

    if let Some(file) = input.strip_prefix("\\i ") {
        return Command::ExecuteFile(file.trim().to_string());
    }

    if let Some(name) = input.strip_prefix("\\create ") {
        return Command::CreateCollection(name.trim().to_string());
    }

    if let Some(name) = input.strip_prefix("\\use ") {
        return Command::UseCollection(name.trim().to_string());
    }

    if let Some(name) = input.strip_prefix("\\drop ") {
        return Command::DropCollection(name.trim().to_string());
    }

    if input == "\\clear" || input == "\\cls" {
        return Command::Clear;
    }

    if input == "\\help" || input == "\\h" || input == "?" {
        return Command::Help;
    }

    if input == "\\quit" || input == "\\q" || input == "exit" || input == "quit" {
        return Command::Quit;
    }

    Command::Unknown(input.to_string())
}
