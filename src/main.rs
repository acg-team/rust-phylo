use crate::cli::Cli;
use anyhow::Error;
use clap::Parser;
use log::LevelFilter;
use log::{error, info};
use parsimony::parsimony_alignment::pars_align_on_tree;
use parsimony::parsimony_alignment::parsimony_costs::parsimony_costs_model::{
    DNAParsCosts, ProteinParsCosts,
};
use phylo::alignment::{self, Alignment};
use phylo::io;
use phylo::phylo_info::{setup_phylogenetic_info, PhyloInfo};
use phylo::sequences::{get_sequence_type, SequenceType};
use phylo::tree::{get_percentiles, NodeIdx};
use pretty_env_logger::env_logger::Builder;
use std::path::PathBuf;
use std::result::Result::Ok;

mod cli;

type Result<T> = std::result::Result<T, Error>;

fn cmp_f64() -> impl Fn(&f64, &f64) -> std::cmp::Ordering {
    |a, b| a.partial_cmp(b).unwrap()
}

#[cfg(test)]
fn assert_float_relative_slice_eq(actual: &[f64], expected: &[f64], epsilon: f64) {
    use approx::relative_eq;
    assert_eq!(
        actual.len(),
        expected.len(),
        "Must have the same number of entries."
    );
    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            relative_eq!(exp, act, epsilon = epsilon),
            "Entries at position {} do not match",
            i,
        );
    }
}

fn indel_map_align_dna(
    info: &PhyloInfo,
    model_name: String,
    model_params: Vec<f64>,
    go: f64,
    ge: f64,
    categories: u32,
) -> Result<(Vec<Alignment>, Vec<f64>)> {
    let scoring = DNAParsCosts::new(
        &model_name,
        &model_params,
        go,
        ge,
        &get_percentiles(&info.tree.get_all_branch_lengths(), categories),
        false,
        false,
    )?;
    Ok(pars_align_on_tree(&Box::new(&scoring), info))
}

fn indel_map_align_protein(
    info: &PhyloInfo,
    model_name: String,
    _: Vec<f64>,
    go: f64,
    ge: f64,
    categories: u32,
) -> Result<(Vec<Alignment>, Vec<f64>)> {
    let scoring = ProteinParsCosts::new(
        &model_name,
        go,
        ge,
        &get_percentiles(&info.tree.get_all_branch_lengths(), categories),
        false,
        false,
    )?;
    Ok(pars_align_on_tree(&Box::new(&scoring), info))
}

fn main() -> Result<()> {
    Builder::new()
        .filter_level(LevelFilter::Info)
        .format_timestamp_secs()
        .format_module_path(false)
        .init();
    info!("JATI run started");
    let cli = Cli::try_parse()?;
    info!("Successfully parsed the command line parameters");
    let info = setup_phylogenetic_info(cli.seq_file, cli.tree_file);
    match info {
        Ok(info) => {
            match cli.command {
                Some(command) => match command {
                    cli::Commands::IndelMAP { go, ge, categories } => {
                        let (alignment, scores) = match get_sequence_type(&info.sequences) {
                            SequenceType::DNA => {
                                info!("Working on DNA data -- please ensure that data type is inferred correctly.");
                                indel_map_align_dna(
                                    &info,
                                    cli.model,
                                    cli.model_params,
                                    go,
                                    ge,
                                    categories,
                                )?
                            }
                            SequenceType::Protein => {
                                info!("Working on protein data -- please ensure that data type is inferred correctly.");
                                indel_map_align_protein(
                                    &info,
                                    cli.model,
                                    cli.model_params,
                                    go,
                                    ge,
                                    categories,
                                )?
                            }
                        };
                        info!("Final alignment scores are: \n{:?}", scores);
                        let out_msa_path = match cli.output_msa_file {
                            Some(path) => path,
                            None => {
                                let path = PathBuf::from("msa.fasta");
                                info!(
                                    "No output file name provided, writing MSA to default file {}.",
                                    path.display()
                                );
                                path
                            }
                        };
                        io::write_sequences_to_file(
                            &alignment::compile_alignment_representation(
                                &info,
                                &alignment,
                                None::<NodeIdx>,
                            ),
                            out_msa_path,
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
