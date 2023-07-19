#[macro_use]
extern crate assert_float_eq;

use alignment::Alignment;
use anyhow::Error;
// use anyhow::Ok;
use clap::Parser;
use cli::Cli;
use env_logger::Env;
use log::{error, info, trace, warn};
use parsimony_alignment::pars_align_on_tree;
use parsimony_alignment::parsimony_costs::parsimony_costs_model::DNAParsCosts;
use parsimony_alignment::parsimony_costs::parsimony_costs_model::ProteinParsCosts;
use phylo_info::setup_phylogenetic_info;
use phylo_info::PhyloInfo;

use sequences::get_sequence_type;

use std::result::Result::Ok;

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

fn indelMAP_align_DNA(
    info: &PhyloInfo,
    model_name: String,
    model_params: Vec<f64>,
    go: f64,
    ge: f64,
) -> Result<(Vec<Alignment>, Vec<f64>)> {
    let scoring = DNAParsCosts::new(
        &model_name,
        &model_params,
        go,
        ge,
        &[0.1, 0.3, 0.5, 0.7],
        false,
        false,
    )?;
    Ok(pars_align_on_tree(&Box::new(&scoring), info))
}

fn indelMAP_align_protein(
    info: &PhyloInfo,
    model_name: String,
    model_params: Vec<f64>,
    go: f64,
    ge: f64,
) -> Result<(Vec<Alignment>, Vec<f64>)> {
    let scoring = ProteinParsCosts::new(&model_name, go, ge, &[0.1, 0.3, 0.5, 0.7], false, false)?;
    Ok(pars_align_on_tree(&Box::new(&scoring), info))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("JATI run started");
    let cli = Cli::try_parse()?;
    info!("Successfully parsed the command line parameters");
    let info = setup_phylogenetic_info(cli.seq_file, cli.tree_file);
    match info {
        Ok(info) => {
            match cli.command {
                Some(command) => match command {
                    cli::Commands::IndelMAP { go, ge } => {
                        let (alignment, scores) = match get_sequence_type(&info.sequences) {
                            crate::sequences::SequenceType::DNA => {
                                info!("Working on DNA data -- please ensure that data type is inferred correctly.");
                                indelMAP_align_DNA(&info, cli.model, cli.model_params, go, ge)?
                            }
                            crate::sequences::SequenceType::Protein => {
                                info!("Working on DNA data -- please ensure that data type is inferred correctly.");
                                indelMAP_align_protein(&info, cli.model, cli.model_params, go, ge)?
                            }
                        };
                        info!("Final alignment scores are: \n{:?}", scores);
                        io::write_sequences_to_file(
                            &alignment::compile_alignment_representation(&info, &alignment, None),
                            cli.output_msa_file,
                        )?;
                        info!("IndelMAP alignment done, quitting.");
                    }
                },
                None => {
                    info!("Can only do IndelMAP alignments at the moment, quitting.");
                    return Ok(());
                }
            };
        }
        Err(error) => {
            error!("Error in input data: {}", error);
            return Ok(());
        }
    }
    Ok(())
}
