use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn test_version() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("consilium")?;
    cmd.arg("--version-flag");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("consilium 0.1.4"));
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
