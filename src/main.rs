#[macro_use]
extern crate assert_float_eq;

use anyhow::Error;
use clap::Parser;
use cli::Cli;
use env_logger::Env;
use log::{error, info, trace, warn};
use phylo_info::setup_phylogenetic_info;
use phylo_info::PhyloInfo;

mod alignment;
mod cli;
mod io;
mod parsimony_alignment;
mod phylo_info;
mod sequences;
mod substitution_models;
mod tree;

type Result<T> = std::result::Result<T, Error>;

#[allow(non_camel_case_types)]
type f64_h = ordered_float::OrderedFloat<f64>;

fn cmp_f64() -> impl Fn(&f64, &f64) -> std::cmp::Ordering {
    |a, b| a.partial_cmp(b).unwrap()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("JATI run started");
    let cli = Cli::try_parse()?;
    info!("Successfully parsed the command line parameters");
    let info = setup_phylogenetic_info(cli.seq_file, cli.tree_file);
    if let Err(error) = &info {
        error!("Error in input data: {}", error);
        return Ok(());
    }
    println!("{:?}", info.unwrap().sequences);
    Ok(())
}
