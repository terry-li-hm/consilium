use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn test_version() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("consilium"));
    cmd.arg("--version-flag");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("consilium 0.5.3"));
    Ok(())
}

#[test]
fn test_no_args_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("consilium"));
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Error: question is required"));
    Ok(())
}

#[test]
fn test_help_includes_examples() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("consilium"));
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Examples:"))
        .stdout(predicate::str::contains(
            "consilium \"Should I take this job offer?\"",
        ))
        .stdout(predicate::str::contains("--judge-model"))
        .stdout(predicate::str::contains("--critic-model"))
        .stdout(predicate::str::contains("--no-critic"));
    Ok(())
}
