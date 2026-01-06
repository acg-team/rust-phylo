use pest::error::Error as PestError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    // Io(#[from] std::io::Error),
    Io(String),

    #[error("Alphabet error: {0}")]
    Alphabet(String),

    #[error("Sequence error: {0}")]
    Sequence(String),

    #[error("Alignment error: {0}")]
    Alignment(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Ancestral alignment error: {0}")]
    AncestralAlignment(String),

    #[error("Tree error: {0}")]
    Tree(String),

    #[error("Tree move error: {0}")]
    TreeMove(String),

    #[error("Tree parsing error: {0}")]
    TreeParsing(String, Box<PestError<crate::tree::tree_parser::Rule>>),

    // Wrapper for external errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}
