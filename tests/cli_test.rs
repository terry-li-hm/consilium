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
        .stdout(predicate::str::contains("--no-critic"))
        .stdout(predicate::str::contains("--web-search"))
        .stdout(predicate::str::contains("--web-engine"));
    Ok(())
}

#[test]
fn test_web_search_zero_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("consilium"));
    cmd.args(["test question", "--web-search", "0"]);
    cmd.assert().failure();
    Ok(())
}

#[test]
fn test_web_engine_requires_web_search() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("consilium"));
    cmd.args(["test question", "--web-engine", "native"]);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--web-search"));
    Ok(())
}
