use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn test_version() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.arg("--version-flag");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("consilium 0.2.0"));
    Ok(())
}

#[test]
fn test_list_roles() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.arg("--list-roles");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Advocate"))
        .stdout(predicate::str::contains("Skeptic"))
        .stdout(predicate::str::contains("Pragmatist"));
    Ok(())
}

#[test]
fn test_no_args_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Error: question is required"));
    Ok(())
}

#[test]
fn test_help_includes_examples() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Examples:"))
        .stdout(predicate::str::contains(
            "consilium \"Should I take this job offer?\"",
        ));
    Ok(())
}

#[test]
fn test_help_includes_cc_flag() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--cc"))
        .stdout(predicate::str::contains("Claude Code-friendly mode"));
    Ok(())
}

#[test]
fn test_cc_flag_no_question_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.arg("--cc");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Error: question is required"))
        .stderr(predicate::str::contains("unexpected argument").not());
    Ok(())
}
