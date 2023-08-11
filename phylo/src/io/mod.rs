use crate::tree::{tree_parser, Tree};
use crate::Result;
use anyhow::bail;
use bio::{alphabets, io::fasta};
use log::info;
use std::{error::Error, fmt, fs, path::PathBuf};

#[derive(Debug)]
pub(crate) struct DataError {
    pub(crate) message: String,
}
impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl Error for DataError {}

pub fn read_sequences_from_file(path: PathBuf) -> Result<Vec<fasta::Record>> {
    info!("Reading sequences from file {}.", path.display());
    let reader = fasta::Reader::from_file(path)?;
    let mut sequences = Vec::new();
    let mut alphabet = alphabets::protein::iupac_alphabet();
    alphabet.insert('-' as u8);
    for result in reader.records() {
        let rec = result?;
        if let Err(e) = rec.check() {
            bail!(DataError {
                message: e.to_string()
            });
        }
        if !alphabet.is_word(rec.seq()) {
            bail!(DataError {
                message: String::from("Invalid genetic sequences")
            });
        }
        sequences.push(rec);
    }
    info!("Read sequences successfully.");
    Ok(sequences)
}

pub fn write_sequences_to_file(sequences: &[fasta::Record], path: PathBuf) -> Result<()> {
    info!("Writing sequences/MSA to file {}.", path.display());
    let mut writer = fasta::Writer::to_file(path)?;
    for rec in sequences {
        writer.write_record(rec)?;
    }
    info!("Finished writing successfully.");
    Ok(())
}

// Currently parsing only rooted trees
pub fn read_newick_from_file(path: PathBuf) -> Result<Vec<Tree>> {
    info!("Reading rooted newick tree from file {}.", path.display());
    let newick = fs::read_to_string(path)?;
    info!("Read file successfully.");
    tree_parser::from_newick_string(&newick)
}

#[cfg(test)]
mod io_tests;
