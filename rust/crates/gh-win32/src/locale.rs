//! Windows UI locale detector implementing `gh_core::traits::LocaleProvider` (`src/i18n.py`).

use gh_core::traits::LocaleProvider;
use windows::Win32::Globalization::{GetUserDefaultLocaleName, GetUserDefaultUILanguage};

/// Maximum locale name length (`LOCALE_NAME_MAX_LENGTH` in `winnls.h`).
const LOCALE_NAME_MAX_LENGTH: usize = 85;

/// Primary language ID constants from `winnt.h`.
const LANG_ENGLISH: u16 = 0x09;
const LANG_VIETNAMESE: u16 = 0x2A;

/// Maps a Win32 `LANGID` (low 10-bit primary language ID) to a canonical BCP-47 tag
/// supported by Genshin Helper (`"vi-VN"`, `"en-US"`), or `""` for other languages.
pub fn lang_id_to_tag(lang_id: u16) -> &'static str {
    match lang_id & 0x03FF {
        LANG_VIETNAMESE => "vi-VN",
        LANG_ENGLISH => "en-US",
        _ => "",
    }
}

/// Queries the active Windows user locale (`GetUserDefaultLocaleName`), falling back to
/// `GetUserDefaultUILanguage` (`0x2A -> "vi-VN"`, `0x09 -> "en-US"`), matching
/// `src/i18n.py::_detect_windows_locale_name`.
pub fn detect_windows_locale_name() -> Option<String> {
    let mut buf = [0u16; LOCALE_NAME_MAX_LENGTH];
    // SAFETY: `buf` is a stack-allocated writable buffer of `LOCALE_NAME_MAX_LENGTH` UTF-16 code units.
    let len = unsafe { GetUserDefaultLocaleName(&mut buf) };
    if len > 0 {
        let max_len = (len as usize).min(buf.len());
        let slice = &buf[..max_len];
        let nul_pos = slice.iter().position(|&c| c == 0).unwrap_or(slice.len());
        let tag = String::from_utf16_lossy(&slice[..nul_pos])
            .trim()
            .to_owned();
        if !tag.is_empty() {
            return Some(tag);
        }
    }

    // SAFETY: Pure Win32 getter with no pointer arguments.
    let lang_id = unsafe { GetUserDefaultUILanguage() };
    let mapped = lang_id_to_tag(lang_id);
    if mapped.is_empty() {
        None
    } else {
        Some(mapped.to_owned())
    }
}

/// Windows implementation of `gh_core::traits::LocaleProvider`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Win32LocaleProvider;

impl Win32LocaleProvider {
    pub const fn new() -> Self {
        Self
    }
}

impl LocaleProvider for Win32LocaleProvider {
    fn system_language_tag(&self) -> String {
        detect_windows_locale_name().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gh_core::i18n::{normalize_lang_code, resolve_language};

    struct MockLocale(&'static str);

    impl LocaleProvider for MockLocale {
        fn system_language_tag(&self) -> String {
            self.0.to_owned()
        }
    }

    #[test]
    fn lang_id_to_tag_maps_vietnamese_english_and_unknown() {
        assert_eq!(lang_id_to_tag(0x042A), "vi-VN");
        assert_eq!(lang_id_to_tag(0x002A), "vi-VN");
        assert_eq!(lang_id_to_tag(0x0409), "en-US");
        assert_eq!(lang_id_to_tag(0x0809), "en-US");
        assert_eq!(lang_id_to_tag(0x0009), "en-US");
        assert_eq!(lang_id_to_tag(0x040C), "");
        assert_eq!(lang_id_to_tag(0x0411), "");
        assert_eq!(lang_id_to_tag(0x0000), "");
    }

    #[test]
    fn win32_locale_provider_returns_valid_tag_and_resolves() {
        let provider = Win32LocaleProvider::new();
        let tag = provider.system_language_tag();
        assert!(!tag.contains('\0'));
        assert!(!tag.is_empty(), "expected non-empty Windows locale tag");

        let resolved = resolve_language(Some("auto"), Some(&provider));
        assert!(resolved == "vi" || resolved == "en");

        assert_eq!(resolve_language(Some("vi"), Some(&provider)), "vi");
        assert_eq!(resolve_language(Some("en"), Some(&provider)), "en");
    }

    #[test]
    fn resolve_language_respects_priority_and_fallbacks() {
        let vi_prov = MockLocale("vi-VN");
        let en_prov = MockLocale("en-US");
        let fr_prov = MockLocale("fr-FR");
        let empty_prov = MockLocale("");

        assert_eq!(resolve_language(Some("vi"), Some(&en_prov)), "vi");
        assert_eq!(resolve_language(Some("en"), Some(&vi_prov)), "en");
        assert_eq!(resolve_language(Some("auto"), Some(&vi_prov)), "vi");
        assert_eq!(resolve_language(Some(""), Some(&vi_prov)), "vi");
        assert_eq!(resolve_language(None, Some(&en_prov)), "en");
        assert_eq!(normalize_lang_code(Some("fr-FR")), None);

        // Unknown OS locale falls back to env or default "en"
        if std::env::var("LC_ALL").is_err()
            && std::env::var("LC_MESSAGES").is_err()
            && std::env::var("LANG").is_err()
        {
            assert_eq!(resolve_language(Some("auto"), Some(&fr_prov)), "en");
            assert_eq!(resolve_language(None, Some(&empty_prov)), "en");
        }
    }
}
