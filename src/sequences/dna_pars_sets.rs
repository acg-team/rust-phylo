
pub(crate) fn dna_pars_sets() -> [u8; 256] {
    let possible_upper_chars = [
        b'T', b'C', b'A', b'G', b'X', b'N', b'V', b'D', b'B', b'H', b'M', b'R', b'W', b'S', b'Y',
        b'K',
    ];
    let mut pars_table = [0b11110 as u8; 256]; //u8::MAX as usize];
    pars_table['T' as usize] = 0b00001;
    pars_table['C' as usize] = 0b00010;
    pars_table['A' as usize] = 0b00100;
    pars_table['G' as usize] = 0b01000;
    pars_table['-' as usize] = 0b10000;
    pars_table['X' as usize] = 0b01111;
    pars_table['N' as usize] = 0b01111;
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
    for ch in possible_upper_chars {
        pars_table[ch.to_ascii_lowercase() as usize] = pars_table[ch as usize];
    }
    pars_table
}


