use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serializer};

pub const POSITIVE_INFINITY_TOKEN: &str = "Infinity";
pub const NEGATIVE_INFINITY_TOKEN: &str = "-Infinity";

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ProfitFactorWire {
    Number(f64),
    String(String),
    Null(()),
}

fn parse_profit_factor_string(token: &str) -> Result<f64, String> {
    match token.trim().to_ascii_lowercase().as_str() {
        "infinity" | "+infinity" | "inf" | "+inf" => Ok(f64::INFINITY),
        "-infinity" | "-inf" => Ok(f64::NEG_INFINITY),
        other => Err(format!("unsupported profit_factor token {other:?}")),
    }
}

fn parse_profit_factor_wire(wire: ProfitFactorWire) -> Result<f64, String> {
    match wire {
        ProfitFactorWire::Number(value) => Ok(value),
        ProfitFactorWire::String(token) => parse_profit_factor_string(&token),
        // Legacy artefacts serialised non-finite floats as JSON null.
        ProfitFactorWire::Null(()) => Ok(f64::INFINITY),
    }
}

pub fn deserialize_profit_factor<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let wire = ProfitFactorWire::deserialize(deserializer)?;
    parse_profit_factor_wire(wire).map_err(D::Error::custom)
}

pub fn serialize_profit_factor<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_finite() {
        serializer.serialize_f64(*value)
    } else if value.is_infinite() && value.is_sign_positive() {
        serializer.serialize_str(POSITIVE_INFINITY_TOKEN)
    } else if value.is_infinite() && value.is_sign_negative() {
        serializer.serialize_str(NEGATIVE_INFINITY_TOKEN)
    } else {
        Err(serde::ser::Error::custom(
            "profit_factor must be finite or +/-Infinity",
        ))
    }
}

pub fn profit_factor_json_value(value: f64) -> Result<serde_json::Value, String> {
    if value.is_finite() {
        Ok(serde_json::json!(value))
    } else if value.is_infinite() && value.is_sign_positive() {
        Ok(serde_json::json!(POSITIVE_INFINITY_TOKEN))
    } else if value.is_infinite() && value.is_sign_negative() {
        Ok(serde_json::json!(NEGATIVE_INFINITY_TOKEN))
    } else {
        Err("profit_factor must be finite or +/-Infinity".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct ProfitFactorEnvelope {
        #[serde(
            serialize_with = "serialize_profit_factor",
            deserialize_with = "deserialize_profit_factor"
        )]
        profit_factor: f64,
    }

    #[test]
    fn serialises_positive_infinity_as_string_token() {
        let payload = ProfitFactorEnvelope {
            profit_factor: f64::INFINITY,
        };

        let json = serde_json::to_string(&payload).unwrap();

        assert_eq!(json, r#"{"profit_factor":"Infinity"}"#);
    }

    #[test]
    fn deserialises_legacy_null_profit_factor_as_infinity() {
        let payload: ProfitFactorEnvelope =
            serde_json::from_str(r#"{"profit_factor":null}"#).unwrap();

        assert!(payload.profit_factor.is_infinite());
        assert!(payload.profit_factor.is_sign_positive());
    }

    #[test]
    fn profit_factor_json_value_uses_string_token_for_infinity() {
        let value = profit_factor_json_value(f64::INFINITY).unwrap();

        assert_eq!(value, serde_json::Value::String("Infinity".to_string()));
    }
}
