use std::path::PathBuf;
use clap::{Parser};

#[derive(Parser)]
#[command(author, version, about, long_about = None)] 
pub(super) struct Cli {
    /// Sequence file in fasta format
    #[arg(short, long, value_name = "SEQ_FILE")]
    pub(super) seq_file: PathBuf,

    /// Tree file in newick format
    #[arg(short, long, value_name = "TREE_FILE")]
    pub(super) tree_file: PathBuf,

    /// Sequence evolution model
    #[arg(short, long, value_name = "MODEL", rename_all = "UPPER")]
    pub(super) model: String,

    /// Sequence evolution model parameters, e.g. alpha and beta for k80
    #[arg(short = 'p', long, value_name = "MODEL_PARAMS")]
    pub(super) model_params: Vec<f64>,

    /// Gap opening penalty
    #[arg(short = 'o', long, default_value_t = 2.5)]
    pub(super) go: f64,

    /// Gap extension penalty
    #[arg(short = 'e', long, default_value_t = 0.5)]
    pub(super) ge: f64,
}