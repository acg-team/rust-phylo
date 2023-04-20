use crate::sequences::{NUCLEOTIDES_STR, AMB_NUCLEOTIDES_STR, AMINOACIDS_STR, AMB_AMINOACIDS_STR, charify, GAP};

pub(crate) const GAP_SET: u32 = 0b1;
pub(crate) const EMPTY_SET: u32 = 0b0;

pub(crate) fn dna_pars_sets() -> [u32; u8::MAX as usize] {
    let nucleotides = charify(NUCLEOTIDES_STR);
    let ambiguous_nucleotides = charify(AMB_NUCLEOTIDES_STR);
    let mut pars_table = [0b11110 as u32; u8::MAX as usize];
    let mut cur_code = GAP_SET;
    pars_table[GAP as usize] = GAP_SET;
    for &ch in &nucleotides {
        cur_code <<= 1;
        pars_table[ch as usize] = cur_code;
    }
    pars_table['V' as usize] = pars_table['X' as usize] ^ pars_table['T' as usize];
    pars_table['D' as usize] = pars_table['X' as usize] ^ pars_table['C' as usize];
    pars_table['B' as usize] = pars_table['X' as usize] ^ pars_table['A' as usize];
    pars_table['H' as usize] = pars_table['X' as usize] ^ pars_table['G' as usize];
    pars_table['M' as usize] = pars_table['A' as usize] | pars_table['C' as usize];
    pars_table['R' as usize] = pars_table['A' as usize] | pars_table['G' as usize];
    pars_table['W' as usize] = pars_table['A' as usize] | pars_table['T' as usize];
    pars_table['S' as usize] = pars_table['C' as usize] | pars_table['G' as usize];
    pars_table['Y' as usize] = pars_table['C' as usize] | pars_table['T' as usize];
    pars_table['K' as usize] = pars_table['G' as usize] | pars_table['T' as usize];
    for &ch in nucleotides.iter().chain(ambiguous_nucleotides.iter()) {
        pars_table[ch.to_ascii_lowercase() as usize] = pars_table[ch as usize];
    }
    pars_table
}

pub(crate) fn protein_pars_sets() -> [u32; u8::MAX as usize] {
    let aminoacids = charify(AMINOACIDS_STR);
    let ambiguous_acids = charify(AMB_AMINOACIDS_STR);
    let mut pars_table = [0b11111_11111_11111_11111_0 as u32; u8::MAX as usize];
    let mut cur_code = GAP_SET;
    pars_table[GAP as usize] = GAP_SET;
    for &ch in &aminoacids {
        cur_code <<= 1;
        pars_table[ch as usize] = cur_code;
    }
    pars_table['B' as usize] = pars_table['D' as usize] | pars_table['N' as usize];
    pars_table['Z' as usize] = pars_table['E' as usize] | pars_table['Q' as usize];
    pars_table['J' as usize] = pars_table['I' as usize] | pars_table['L' as usize];
    for &ch in aminoacids.iter().chain(ambiguous_acids.iter()) {
        pars_table[ch.to_ascii_lowercase() as usize] = pars_table[ch as usize];
    }
    pars_table
}


#[cfg(test)]
mod parsimony_sets_tests {
    use crate::sequences::{parsimony_sets::{GAP_SET, EMPTY_SET}, GAP};

    use super::{dna_pars_sets, protein_pars_sets};

    #[test]
    fn protein_sets() {
        let pars_table = protein_pars_sets();
        assert_eq!(pars_table['X' as usize], pars_table['O' as usize]);
        assert_eq!(pars_table['X' as usize], !pars_table[GAP as usize] % (0b1 << 21));
        assert_eq!(pars_table['n' as usize], pars_table['N' as usize]);
        assert_eq!(pars_table['e' as usize], pars_table['E' as usize]);
        assert_eq!(pars_table['b' as usize], pars_table['D' as usize] | pars_table['n' as usize]);
        assert_eq!(pars_table['Z' as usize], pars_table['E' as usize] | pars_table['q' as usize]);
        assert_eq!(pars_table['j' as usize], pars_table['i' as usize] | pars_table['L' as usize]);
    }

    #[test]
    fn dna_sets() {
        let pars_table = dna_pars_sets();
        assert_eq!(pars_table['N' as usize], pars_table['X' as usize]);
        assert_eq!(pars_table['N' as usize], pars_table['n' as usize]);
        assert_eq!(pars_table['T' as usize], pars_table['t' as usize]);
        assert_eq!(pars_table['C' as usize], pars_table['c' as usize]);
        assert_eq!(pars_table['A' as usize], pars_table['a' as usize]);
        assert_eq!(pars_table['G' as usize], pars_table['g' as usize]);
        assert_eq!(pars_table['V' as usize] & pars_table['t' as usize], EMPTY_SET);
        assert_eq!(pars_table['D' as usize] & pars_table['c' as usize], EMPTY_SET);
        assert_eq!(pars_table['b' as usize] & pars_table['A' as usize], EMPTY_SET);
        assert_eq!(pars_table['h' as usize] & pars_table['G' as usize], EMPTY_SET);
        assert_eq!(pars_table['m' as usize], pars_table['A' as usize] | pars_table['C' as usize]);
        assert_eq!(pars_table['k' as usize], pars_table['G' as usize] | pars_table['T' as usize]);
        assert_eq!(pars_table['-' as usize], GAP_SET);
    }

}