use std::fs;
use std::path::PathBuf;

use crate::tree::{self, Tree};
use crate::Result;
use anyhow::bail;
use bio::{alphabets, io::fasta};

pub(crate) fn read_sequences_from_file(path: PathBuf) -> Result<Vec<fasta::Record>> {
    let reader = fasta::Reader::from_file(path)?;
    let mut sequences = Vec::new();
    let mut alphabet = alphabets::protein::iupac_alphabet();
    alphabet.insert('-' as u8);
    for result in reader.records() {
        let rec = result?;
        if let Err(e) = rec.check() {
            bail!("{}", e);
        }
        if !alphabet.is_word(rec.seq()) {
            bail!("These are not valid genetic sequences");
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
pub(crate) fn read_newick_from_file(path: PathBuf) -> Result<Vec<Tree>> {
    let newick = fs::read_to_string(path)?;
    tree::from_newick_string(&newick)
}

// Currently parsing only rooted trees
pub(crate) fn read_newick_from_string(newick: &str) -> Result<Vec<Tree>> {
    tree::from_newick_string(newick)
}

#[cfg(test)]
mod io_test {
    use std::path::PathBuf;

    use crate::io::read_sequences_from_file;
    use rstest::*;

    #[test]
    fn reading_correct_fasta() {
        let sequences =
            read_sequences_from_file(PathBuf::from("./data/sequences_DNA1.fasta")).unwrap();
        assert_eq!(sequences.len(), 4);
        for seq in sequences {
            assert_eq!(seq.seq().len(), 5);
        }

        let corr_lengths = vec![1, 2, 2, 4];
        let sequences =
            read_sequences_from_file(PathBuf::from("./data/sequences_DNA2_unaligned.fasta"))
                .unwrap();
        assert_eq!(sequences.len(), 4);
        for (i, seq) in sequences.into_iter().enumerate() {
            assert_eq!(seq.seq().len(), corr_lengths[i]);
        }
    }

    #[rstest]
    #[case::empty_sequence_name("./data/sequences_garbage_empty_name.fasta")]
    #[case::garbage_sequence("./data/sequences_garbage_non-ascii.fasta")]
    #[case::weird_chars("./data/sequences_garbage_weird_symbols.fasta")]
    fn reading_incorrect_fasta(#[case] input: &str) {
        assert!(read_sequences_from_file(PathBuf::from(input)).is_err());
    }

    #[test]
    fn reading_nonexistent_fasta() {
        assert!(
            read_sequences_from_file(PathBuf::from("./data/sequences_nonexistent.fasta")).is_err()
        );
    }
}
