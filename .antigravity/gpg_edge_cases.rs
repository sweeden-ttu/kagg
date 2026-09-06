use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Simple RAII temporary directory for zero-dependency standalone execution
pub struct AutoCleanDir {
    pub path: PathBuf,
}

impl AutoCleanDir {
    pub fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{}_{}", prefix, nanos));
        fs::create_dir_all(&path).expect("Failed to create temp dir");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        }

        Self { path }
    }
}

impl Drop for AutoCleanDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Executes `gpg --auto-key-import <target>` and captures combined output
pub fn run_gpg_auto_key_import(target: &str, gpg_home: Option<&Path>) -> (i32, String) {
    let mut cmd = Command::new("gpg");
    cmd.arg("--auto-key-import").arg(target);

    if let Some(home) = gpg_home {
        cmd.env("GNUPGHOME", home);
    }

    let output = cmd.output().expect("Failed to execute gpg command");
    let exit_code = output.status.code().unwrap_or(-1);
    let combined_output = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    (exit_code, combined_output)
}

fn main() {
    println!("=== GPG Auto-Key-Import Edge Case Runner ===");
    println!("Run tests using: rustc --test gpg_edge_cases.rs -o /tmp/gpg_tests && /tmp/gpg_tests");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_baseline_nonexistent_file() {
        let nonexistent = "/tmp/definitely_nonexistent_test_key_xyz987.asc";
        let (_, output) = run_gpg_auto_key_import(nonexistent, None);

        // Baseline: complains about missing command and inability to open file
        assert!(output.contains("WARNING: no command supplied"));
        assert!(output.contains("can't open"));
    }

    #[test]
    fn test_edge_case_plain_text_file() {
        let temp_dir = AutoCleanDir::new("gpg_test_plain");
        let file_path = temp_dir.path.join("plain_message.txt");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "Standard ASCII plain text without OpenPGP packets.").unwrap();

        let (_, output) = run_gpg_auto_key_import(file_path.to_str().unwrap(), None);

        // Output divergence: reports "no valid OpenPGP data found" instead of "can't open"
        assert!(output.contains("WARNING: no command supplied"));
        assert!(output.contains("no valid OpenPGP data found"));
        assert!(!output.contains("can't open"));
    }

    #[test]
    fn test_edge_case_directory_target() {
        let temp_dir = AutoCleanDir::new("gpg_test_dir");
        let sub_dir = temp_dir.path.join("nested_folder");
        fs::create_dir(&sub_dir).unwrap();

        let (_, output) = run_gpg_auto_key_import(sub_dir.to_str().unwrap(), None);

        // Output divergence: reports filesystem directory error
        assert!(output.contains("Is a directory"));
        assert!(!output.contains("can't open"));
    }

    #[test]
    fn test_edge_case_stdin_dash() {
        let (_, output) = run_gpg_auto_key_import("-", None);

        // Output divergence: reads stdin stream directly
        assert!(output.contains("WARNING: no command supplied"));
        assert!(!output.contains("can't open"));
    }

    #[test]
    fn test_edge_case_command_injection_flag() {
        let (_, output) = run_gpg_auto_key_import("--list-keys", None);

        // Output divergence: --list-keys is recognized as a command, so "no command supplied" warning is eliminated
        assert!(!output.contains("WARNING: no command supplied"));
    }

    #[test]
    fn test_edge_case_valid_openpgp_public_key() {
        let gpg_home = AutoCleanDir::new("gpg_test_home");

        // 1. Generate an isolated ephemeral key
        let keygen = Command::new("gpg")
            .env("GNUPGHOME", &gpg_home.path)
            .args(&[
                "--batch",
                "--passphrase",
                "",
                "--quick-generate-key",
                "EdgeCase Tester <edgecase@example.com>",
                "default",
                "default",
                "never",
            ])
            .output()
            .expect("Failed to generate test key");
        assert!(keygen.status.success());

        // 2. Export public key
        let pubkey_path = gpg_home.path.join("pubkey.asc");
        let export_cmd = Command::new("gpg")
            .env("GNUPGHOME", &gpg_home.path)
            .args(&["--armor", "--export", "edgecase@example.com"])
            .output()
            .expect("Failed to export public key");

        let mut pubkey_file = File::create(&pubkey_path).unwrap();
        pubkey_file.write_all(&export_cmd.stdout).unwrap();

        // 3. Test `gpg --auto-key-import <pubkey.asc>`
        let (_, output) = run_gpg_auto_key_import(pubkey_path.to_str().unwrap(), Some(&gpg_home.path));

        // Output divergence: GPG parses packets and lists the key specifications
        assert!(output.contains("EdgeCase Tester <edgecase@example.com>"));
        assert!(output.contains("pub"));
        assert!(!output.contains("can't open"));
    }
}
