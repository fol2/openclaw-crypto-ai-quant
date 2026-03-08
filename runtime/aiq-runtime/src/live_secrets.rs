use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LiveSecrets {
    pub secret_key: String,
    pub main_address: String,
}

pub fn load_live_secrets(path: &Path) -> Result<LiveSecrets> {
    let path = expand_path(path);
    let metadata = fs::metadata(&path)
        .with_context(|| format!("failed to stat live secrets file: {}", path.display()))?;
    #[cfg(unix)]
    {
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            anyhow::bail!(
                "live secrets permissions too open: {} (expected group/other bits unset)",
                path.display()
            );
        }
    }

    let payload = fs::read_to_string(&path)
        .with_context(|| format!("failed to read live secrets file: {}", path.display()))?;
    let secrets: LiveSecrets = serde_json::from_str(&payload)
        .with_context(|| format!("failed to parse live secrets JSON: {}", path.display()))?;

    validate_secret_key(&secrets.secret_key)?;
    validate_address(&secrets.main_address, "main_address")?;
    Ok(secrets)
}

pub fn expand_path(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    path.to_path_buf()
}

pub fn validate_secret_key(secret_key: &str) -> Result<()> {
    let stripped = secret_key
        .trim()
        .strip_prefix("0x")
        .unwrap_or(secret_key.trim());
    if stripped.len() != 64 || !stripped.chars().all(|ch| ch.is_ascii_hexdigit()) {
        anyhow::bail!("invalid secret_key format; expected a 64-char hex string");
    }
    Ok(())
}

pub fn validate_address(address: &str, field_name: &str) -> Result<()> {
    let trimmed = address.trim();
    if trimmed.len() != 42
        || !trimmed.starts_with("0x")
        || !trimmed[2..].chars().all(|ch| ch.is_ascii_hexdigit())
    {
        anyhow::bail!("invalid {field_name} format; expected a 0x-prefixed 40-char hex string");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn load_live_secrets_rejects_group_readable_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("secrets.json");
        fs::write(
            &path,
            r#"{"secret_key":"0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","main_address":"0x1111111111111111111111111111111111111111"}"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
            let err = load_live_secrets(&path).unwrap_err().to_string();
            assert!(err.contains("permissions too open"));
        }
    }

    #[test]
    fn load_live_secrets_accepts_private_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("secrets.json");
        fs::write(
            &path,
            r#"{"secret_key":"0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","main_address":"0x1111111111111111111111111111111111111111"}"#,
        )
        .unwrap();
        #[cfg(unix)]
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        let secrets = load_live_secrets(&path).unwrap();
        assert_eq!(
            secrets.secret_key,
            "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
        assert_eq!(
            secrets.main_address,
            "0x1111111111111111111111111111111111111111"
        );
    }
}
