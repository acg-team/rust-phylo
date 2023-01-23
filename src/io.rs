use anyhow::anyhow;
use bio::alphabets;
use bio::io::fasta;
use super::Result;

pub(crate) fn read_sequences_from_file(path: &str) -> Result<Vec<fasta::Record>> {
    let reader = fasta::Reader::from_file(path)?;
    let mut sequences = Vec::new();
    let mut alphabet = alphabets::protein::alphabet();
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
