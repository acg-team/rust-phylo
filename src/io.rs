use anyhow::anyhow;
use bio::alphabets;
use bio::io::fasta;
use crate::Result;
use crate::tree::{self, Tree};

pub(crate) fn read_sequences_from_file(path: &str) -> Result<Vec<fasta::Record>> {
    let reader = fasta::Reader::from_file(path)?;
    let mut sequences = Vec::new();
    let mut alphabet = alphabets::protein::iupac_alphabet();
    alphabet.insert('-' as u8);
    for result in reader.records() {
        let rec = result?;
        if let Err(e) = rec.check() {
            return Err(anyhow!("{}", e));
        }
        if !alphabet.is_word(rec.seq()) {
            return Err(anyhow!("These are not valid genetic sequences"));
        }
        sequences.push(rec);
    }
    Ok(sequences)
}

pub(crate) fn write_sequences_to_file(sequences: &[fasta::Record], path: &str) -> Result<()> {
    let mut writer = fasta::Writer::to_file(path)?;
    for rec in sequences {
        writer.write_record(rec)?;
    }
    Ok(())
}

// Currently parsing only rooted trees
pub(crate) fn read_newick_from_string(newick: &str) -> Result<Vec<Tree>> {
    tree::from_newick_string(newick)
}
