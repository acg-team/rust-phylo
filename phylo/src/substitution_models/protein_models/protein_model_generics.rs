use lazy_static::lazy_static;

use crate::frequencies;
use crate::sequences::{AMINOACIDS, GAP};
use crate::substitution_models::{FreqVector, SubstMatrix};

use crate::substitution_models::protein_models::{ProteinFrequencyArray, ProteinSubstArray};

lazy_static! {
    pub static ref AMINOACID_INDEX: [usize; 255] = {
        let mut index = [0; 255];
        for (i, &char) in AMINOACIDS.iter().enumerate() {
            index[char as usize] = i;
            index[char.to_ascii_lowercase() as usize] = i;
        }
        index[GAP as usize] = 20;
        index
    };
    pub static ref PROTEIN_GAP_SETS: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 21]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            if char == GAP {
                elem.set_column(0, &frequencies!(&[0.0; 20]).resize_vertically(21, 1.0));
            } else {
                elem.set_column(0, &generic_protein_sets(char).resize_vertically(21, 0.0));
            }
        }
        map
    };
    pub static ref PROTEIN_SETS: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &generic_protein_sets(char));
        }
        map
    };
}

fn generic_protein_sets(char: u8) -> FreqVector {
    let index = &AMINOACID_INDEX;
    if AMINOACIDS.contains(&char.to_ascii_uppercase()) {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index[char as usize], 1.0);
        set
    } else if char.to_ascii_uppercase() == b'B' {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index['D' as usize], 0.5);
        set.fill_row(index['N' as usize], 0.5);
        set
    } else if char.to_ascii_uppercase() == b'Z' {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index['E' as usize], 0.5);
        set.fill_row(index['Q' as usize], 0.5);
        set
    } else if char.to_ascii_uppercase() == b'J' {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index['I' as usize], 0.5);
        set.fill_row(index['L' as usize], 0.5);
        set
    } else {
        frequencies!(&[1.0 / 20.0; 20])
    }
}

pub fn wag_q() -> SubstMatrix {
    SubstMatrix::from_row_slice(20, 20, &WAG_ARR)
}

pub fn wag_freqs() -> FreqVector {
    frequencies!(&WAG_PI_ARR)
}

pub fn blosum_q() -> SubstMatrix {
    SubstMatrix::from_row_slice(20, 20, &BLOSUM_ARR)
}
pub fn blosum_freqs() -> FreqVector {
    frequencies!(&BLOSUM_PI_ARR)
}

pub fn hivb_q() -> SubstMatrix {
    SubstMatrix::from_row_slice(20, 20, &HIVB_ARR)
}

pub fn hivb_freqs() -> FreqVector {
    frequencies!(&HIVB_PI_ARR)
}

const WAG_ARR: ProteinSubstArray = [
    -1.11715057e+00,
    2.54545984e-02,
    2.09164670e-02,
    4.42435753e-02,
    2.08117574e-02,
    3.50234436e-02,
    9.64488758e-02,
    1.23784499e-01,
    8.12702170e-03,
    9.83413728e-03,
    3.60024059e-02,
    5.89977967e-02,
    1.82884104e-02,
    8.49024422e-03,
    6.90921971e-02,
    2.45933079e-01,
    1.35822601e-01,
    1.70810650e-03,
    8.91220187e-03,
    1.49259156e-01,
    5.01473302e-02,
    -9.74317971e-01,
    2.60650109e-02,
    8.81904364e-03,
    1.07031693e-02,
    1.17008471e-01,
    2.67594522e-02,
    5.10845220e-02,
    5.47986914e-02,
    9.51083433e-03,
    4.50280923e-02,
    3.48377118e-01,
    1.39832154e-02,
    4.14283292e-03,
    3.26352146e-02,
    8.93169898e-02,
    3.55011364e-02,
    1.75731159e-02,
    1.41246562e-02,
    1.87390737e-02,
    4.63539889e-02,
    2.93207534e-02,
    -1.45243780e+00,
    3.25057649e-01,
    5.37510081e-03,
    5.95022094e-02,
    5.77162600e-02,
    9.83446839e-02,
    1.01443284e-01,
    2.81916513e-02,
    1.19003416e-02,
    1.96081669e-01,
    4.05726158e-03,
    3.87868402e-03,
    9.36955608e-03,
    2.89960105e-01,
    1.29992329e-01,
    1.08581389e-03,
    4.02045869e-02,
    1.46018775e-02,
    6.71876816e-02,
    6.79797190e-03,
    2.22741453e-01,
    -9.89664514e-01,
    6.13890511e-04,
    2.37749418e-02,
    3.76214291e-01,
    7.56295399e-02,
    2.38634756e-02,
    2.00599411e-03,
    7.67292822e-03,
    3.12385315e-02,
    2.12367569e-03,
    1.88486374e-03,
    2.03635509e-02,
    7.81956861e-02,
    2.40040710e-02,
    1.95925023e-03,
    1.20580812e-02,
    1.13346362e-02,
    9.33756742e-02,
    2.43756285e-02,
    1.08821029e-02,
    1.81374603e-03,
    -4.87356833e-01,
    3.80910275e-03,
    1.30105594e-03,
    2.67953353e-02,
    6.38389434e-03,
    8.65405098e-03,
    3.47693768e-02,
    4.81960241e-03,
    7.99253166e-03,
    1.60540776e-02,
    5.25457073e-03,
    1.02702974e-01,
    3.28482827e-02,
    1.08264780e-02,
    2.01331318e-02,
    7.45652169e-02,
    8.26072508e-02,
    1.40086106e-01,
    6.33276882e-02,
    3.69266021e-02,
    2.00242850e-03,
    -1.37978519e+00,
    3.33274936e-01,
    2.88379648e-02,
    1.10105331e-01,
    5.79447806e-03,
    7.86693035e-02,
    2.53557754e-01,
    3.16289598e-02,
    4.03029062e-03,
    4.48289751e-02,
    7.50664287e-02,
    5.49363362e-02,
    3.25724389e-03,
    8.43000597e-03,
    2.24171105e-02,
    1.43908403e-01,
    2.02667745e-02,
    3.88587103e-02,
    3.69644996e-01,
    4.32673163e-04,
    2.10829953e-01,
    -1.23689359e+00,
    4.96037074e-02,
    1.46160186e-02,
    6.48004717e-03,
    1.39573505e-02,
    1.68246237e-01,
    6.45007592e-03,
    3.27252380e-03,
    3.27728658e-02,
    5.14323998e-02,
    5.26847179e-02,
    2.36373145e-03,
    7.26729376e-03,
    4.38051118e-02,
    1.28804316e-01,
    2.69818623e-02,
    4.61759948e-02,
    5.18222389e-02,
    6.21438786e-03,
    1.27224115e-02,
    3.45930861e-02,
    -4.97615305e-01,
    6.39512510e-03,
    1.54886836e-03,
    5.54661345e-03,
    2.43186032e-02,
    3.56354393e-03,
    2.01395947e-03,
    1.16984369e-02,
    9.78992830e-02,
    1.44609310e-02,
    5.08784222e-03,
    3.83550278e-03,
    1.39322980e-02,
    2.88165928e-02,
    9.86279100e-02,
    1.62306431e-01,
    5.57192762e-02,
    5.04512470e-03,
    1.65523718e-01,
    3.47337211e-02,
    2.17919503e-02,
    -9.92342352e-01,
    7.02914336e-03,
    4.51901377e-02,
    5.79670693e-02,
    8.27210918e-03,
    2.74023283e-02,
    3.34377320e-02,
    5.40027831e-02,
    3.03076161e-02,
    3.96432356e-03,
    1.43397841e-01,
    8.80654393e-03,
    1.75774906e-02,
    8.62894415e-03,
    2.27374806e-02,
    2.36108065e-03,
    3.44758564e-03,
    4.39112305e-03,
    7.76264619e-03,
    2.66054716e-03,
    3.54333162e-03,
    -1.23316235e+00,
    2.86901849e-01,
    2.10814436e-02,
    8.71432840e-02,
    4.27335650e-02,
    4.79948583e-03,
    2.33063652e-02,
    9.33714344e-02,
    3.20811429e-03,
    1.55550288e-02,
    5.81951554e-01,
    3.61773460e-02,
    2.29671528e-02,
    5.39592403e-03,
    5.07723042e-03,
    7.78712400e-03,
    3.35159212e-02,
    9.39981231e-03,
    5.35634973e-03,
    1.28067117e-02,
    1.61293890e-01,
    -7.26011058e-01,
    1.67668149e-02,
    9.93538973e-02,
    8.53150677e-02,
    1.99725943e-02,
    2.51521821e-02,
    2.09148274e-02,
    1.00449792e-02,
    1.47571566e-02,
    1.33956077e-01,
    8.23951408e-02,
    2.46964120e-01,
    1.23567432e-01,
    2.87287663e-02,
    1.50020989e-03,
    1.50135495e-01,
    1.57478831e-01,
    3.26392582e-02,
    2.28315787e-02,
    1.64719701e-02,
    2.33029658e-02,
    -1.12458038e+00,
    1.91231107e-02,
    3.58318686e-03,
    2.67471886e-02,
    7.05618738e-02,
    8.88135130e-02,
    2.07608023e-03,
    4.93353966e-03,
    2.27261186e-02,
    8.12342182e-02,
    3.15274268e-02,
    8.13199818e-03,
    6.21171899e-03,
    7.91265839e-03,
    5.95646550e-02,
    1.92016650e-02,
    1.52118141e-02,
    1.03625847e-02,
    2.16559061e-01,
    4.39180223e-01,
    6.08213111e-02,
    -1.32209849e+00,
    4.80238841e-02,
    8.22876997e-03,
    3.60353441e-02,
    9.70828298e-02,
    7.78624074e-03,
    1.58610797e-02,
    1.53161006e-01,
    1.91375401e-02,
    4.74003756e-03,
    3.94504125e-03,
    2.79773419e-03,
    8.06540709e-03,
    3.85161590e-03,
    4.94378712e-03,
    4.36267139e-03,
    1.74197608e-02,
    5.38907773e-02,
    1.91375567e-01,
    5.78321823e-03,
    2.43702603e-02,
    -7.13669467e-01,
    7.75400276e-03,
    3.98311647e-02,
    1.10075915e-02,
    2.30948356e-02,
    2.38942598e-01,
    4.83558564e-02,
    1.30789041e-01,
    3.13579206e-02,
    8.00317997e-03,
    2.53837873e-02,
    2.21694336e-03,
    3.59783992e-02,
    4.15784014e-02,
    2.12816861e-02,
    1.78512221e-02,
    5.08295724e-03,
    3.76245793e-02,
    3.62538959e-02,
    3.50682607e-03,
    6.51181975e-03,
    -6.05589533e-01,
    1.17705024e-01,
    5.09314101e-02,
    2.10476684e-03,
    7.99819538e-03,
    2.34294784e-02,
    3.06463029e-01,
    5.64954735e-02,
    1.63042418e-01,
    6.41659304e-02,
    2.85245740e-02,
    3.96595308e-02,
    4.29545276e-02,
    1.17240186e-01,
    1.89786831e-02,
    1.62485676e-02,
    3.11911675e-02,
    6.29601045e-02,
    1.01094323e-02,
    2.20200458e-02,
    7.74843139e-02,
    -1.39223882e+00,
    2.80340983e-01,
    7.90757000e-03,
    2.91351091e-02,
    1.73171753e-02,
    1.92845533e-01,
    2.55857546e-02,
    8.32830236e-02,
    2.24431082e-02,
    1.03950173e-02,
    3.30702829e-02,
    5.01340994e-02,
    1.97319334e-02,
    1.21360710e-02,
    7.41704586e-02,
    2.95519843e-02,
    9.02923141e-02,
    3.10325114e-02,
    6.93368196e-03,
    3.82015419e-02,
    3.19420652e-01,
    -1.15497297e+00,
    1.67384865e-03,
    1.07785314e-02,
    1.03292625e-01,
    1.02857436e-02,
    5.37140570e-02,
    2.95037596e-03,
    7.76910903e-03,
    1.45305800e-02,
    8.31594682e-03,
    9.53959418e-03,
    2.94435540e-02,
    6.73253520e-03,
    1.08081154e-02,
    6.01955812e-02,
    8.95156718e-03,
    1.05556633e-02,
    6.16978021e-02,
    6.69549041e-03,
    3.82122537e-02,
    7.09903625e-03,
    -4.66693769e-01,
    9.20111217e-02,
    2.71856415e-02,
    2.18869693e-02,
    1.76074690e-02,
    4.45530495e-02,
    1.95002140e-02,
    1.10201360e-02,
    8.77746631e-03,
    1.19614642e-02,
    9.05229631e-03,
    9.93189262e-02,
    2.13722785e-02,
    3.60660117e-02,
    8.67547834e-03,
    8.76940880e-03,
    2.60332425e-01,
    1.03764852e-02,
    5.74190655e-02,
    1.86432946e-02,
    3.75249558e-02,
    -7.26275191e-01,
    2.34177966e-02,
    1.82380955e-01,
    1.16226472e-02,
    8.05097399e-03,
    9.12024801e-03,
    2.03071882e-02,
    1.16133847e-02,
    3.58735465e-02,
    1.63605201e-02,
    3.03481904e-03,
    3.97836594e-01,
    1.62890495e-01,
    1.98837349e-02,
    4.21331247e-02,
    2.62132973e-02,
    1.51237250e-02,
    1.69806541e-02,
    8.88935552e-02,
    5.51642019e-03,
    1.16515558e-02,
    -1.08548744e+00,
];

pub(crate) const WAG_PI_ARR: ProteinFrequencyArray = [
    0.0866279, 0.0439720, 0.0390894, 0.0570451, 0.0193078, 0.0367281, 0.0580589, 0.0832518,
    0.0244313, 0.0484660, 0.0862090, 0.0620286, 0.0195027, 0.0384319, 0.0457631, 0.0695179,
    0.0610127, 0.0143859, 0.0352742, 0.0708956,
];

const BLOSUM_ARR: ProteinSubstArray = [
    -10.98, 0.444, 0.216, 0.218, 0.249, 0.38, 0.544, 1.292, 0.164, 0.375, 0.766, 0.505, 0.38,
    0.236, 0.502, 2.196, 0.759, 0.045, 0.21, 1.499, 0.624, -10.442, 0.53, 0.311, 0.084, 1.053,
    0.831, 0.452, 0.357, 0.25, 0.559, 2.675, 0.201, 0.135, 0.274, 0.733, 0.637, 0.118, 0.264,
    0.354, 0.434, 0.757, -11.567, 1.468, 0.124, 0.726, 0.735, 1.092, 0.656, 0.266, 0.336, 0.961,
    0.178, 0.211, 0.342, 1.57, 0.946, 0.051, 0.296, 0.418, 0.368, 0.374, 1.237, -8.885, 0.087,
    0.545, 1.914, 0.737, 0.23, 0.13, 0.276, 0.574, 0.043, 0.132, 0.31, 0.992, 0.496, 0.046, 0.166,
    0.228, 0.66, 0.158, 0.164, 0.136, -4.929, 0.11, 0.092, 0.284, 0.15, 0.297, 0.476, 0.168, 0.132,
    0.307, 0.112, 0.545, 0.366, 0.048, 0.184, 0.54, 0.846, 1.67, 0.807, 0.718, 0.092, -14.429,
    2.25, 0.673, 0.508, 0.211, 0.739, 1.782, 0.515, 0.183, 0.391, 1.155, 0.914, 0.135, 0.335,
    0.505, 0.769, 0.835, 0.518, 1.6, 0.049, 1.426, -10.73, 0.433, 0.312, 0.237, 0.398, 1.222, 0.13,
    0.117, 0.525, 0.799, 0.65, 0.072, 0.19, 0.448, 1.251, 0.311, 0.527, 0.422, 0.104, 0.293, 0.296,
    -6.188, 0.161, 0.109, 0.291, 0.325, 0.089, 0.189, 0.261, 0.886, 0.291, 0.055, 0.126, 0.201,
    0.412, 0.639, 0.823, 0.343, 0.142, 0.573, 0.555, 0.418, -7.793, 0.164, 0.337, 0.551, 0.112,
    0.24, 0.184, 0.614, 0.618, 0.093, 0.703, 0.272, 0.473, 0.224, 0.167, 0.097, 0.141, 0.12, 0.212,
    0.142, 0.082, -13.276, 3.432, 0.258, 0.804, 0.609, 0.143, 0.288, 0.59, 0.069, 0.239, 5.186,
    0.604, 0.314, 0.132, 0.129, 0.142, 0.261, 0.222, 0.237, 0.106, 2.145, -9.48, 0.257, 1.206,
    1.015, 0.184, 0.242, 0.418, 0.192, 0.271, 1.403, 0.734, 2.766, 0.697, 0.493, 0.092, 1.162,
    1.257, 0.488, 0.319, 0.298, 0.473, -11.821, 0.218, 0.133, 0.424, 0.846, 0.724, 0.077, 0.232,
    0.388, 1.309, 0.494, 0.306, 0.089, 0.172, 0.797, 0.317, 0.317, 0.154, 2.197, 5.271, 0.518,
    -16.524, 0.941, 0.195, 0.593, 0.852, 0.127, 0.366, 1.509, 0.396, 0.161, 0.177, 0.131, 0.194,
    0.138, 0.139, 0.327, 0.16, 0.81, 2.16, 0.153, 0.458, -9.263, 0.214, 0.447, 0.327, 0.354, 1.859,
    0.658, 0.903, 0.35, 0.307, 0.329, 0.076, 0.316, 0.667, 0.485, 0.131, 0.203, 0.419, 0.525,
    0.101, 0.229, -7.339, 0.94, 0.589, 0.087, 0.193, 0.489, 2.432, 0.578, 0.868, 0.65, 0.228,
    0.574, 0.626, 1.013, 0.271, 0.253, 0.34, 0.645, 0.191, 0.295, 0.579, -13.107, 2.75, 0.099,
    0.253, 0.462, 1.016, 0.608, 0.632, 0.393, 0.185, 0.549, 0.616, 0.402, 0.329, 0.626, 0.71,
    0.667, 0.331, 0.261, 0.439, 3.325, -12.88, 0.137, 0.238, 1.416, 0.215, 0.404, 0.123, 0.131,
    0.087, 0.292, 0.243, 0.272, 0.077, 0.263, 1.167, 0.256, 0.177, 1.012, 0.233, 0.43, 0.49,
    -7.698, 1.381, 0.445, 0.441, 0.394, 0.311, 0.207, 0.146, 0.316, 0.282, 0.274, 0.587, 0.398,
    0.722, 0.336, 0.223, 2.326, 0.226, 0.479, 0.373, 0.604, -9.08, 0.435, 1.585, 0.267, 0.221,
    0.143, 0.215, 0.24, 0.335, 0.22, 0.114, 4.345, 1.881, 0.282, 0.463, 0.415, 0.287, 0.441, 1.118,
    0.098, 0.219, -12.889,
];

pub(crate) const BLOSUM_PI_ARR: ProteinFrequencyArray = [
    0.0756, 0.0538, 0.0377, 0.0447, 0.0285, 0.0339, 0.0535, 0.078, 0.03, 0.0599, 0.0958, 0.052,
    0.0219, 0.045, 0.042, 0.0682, 0.0564, 0.0157, 0.036, 0.0715,
];

const HIVB_ARR: ProteinSubstArray = [
    -1.01766542e+00,
    1.07746297e-02,
    1.17064419e-04,
    3.25081940e-02,
    1.31822873e-03,
    1.56751721e-03,
    5.62491150e-02,
    8.19221563e-02,
    1.00259834e-03,
    1.84984813e-04,
    1.12896208e-02,
    1.51128048e-04,
    1.95759980e-04,
    2.15935616e-04,
    5.18225902e-02,
    6.62188070e-02,
    4.53004105e-01,
    8.75747848e-05,
    7.52089070e-05,
    2.48960205e-01,
    9.86921640e-03,
    -1.30086755e+00,
    6.91951394e-03,
    1.11708936e-04,
    3.74641418e-03,
    9.73142382e-02,
    2.84489482e-03,
    1.40163018e-01,
    1.06934771e-01,
    2.50576358e-02,
    3.67880332e-02,
    6.18113716e-01,
    2.63673883e-02,
    7.64271058e-05,
    3.13438064e-02,
    9.34107971e-02,
    8.16370979e-02,
    1.73632424e-02,
    1.49188299e-04,
    2.65643914e-03,
    1.60471410e-04,
    1.03554274e-02,
    -1.37205286e+00,
    3.94582771e-01,
    9.16727006e-04,
    1.91144903e-02,
    3.00974819e-03,
    1.24071385e-02,
    9.04388741e-02,
    2.51788378e-02,
    2.62237077e-04,
    2.38916215e-01,
    5.24423579e-05,
    7.64271058e-05,
    1.80602155e-04,
    3.52923717e-01,
    1.95981341e-01,
    8.75747848e-05,
    2.65362595e-02,
    8.71557550e-04,
    4.66984642e-02,
    1.75193243e-04,
    4.13499625e-01,
    -1.06286594e+00,
    5.32583238e-05,
    1.42209905e-04,
    4.02012104e-01,
    1.08880936e-01,
    2.26740360e-02,
    6.54076700e-04,
    4.59464533e-04,
    1.51128048e-04,
    5.24423579e-05,
    7.64271058e-05,
    8.36757899e-04,
    1.41825143e-02,
    7.81811498e-03,
    8.75747848e-05,
    1.01479829e-02,
    3.42636293e-02,
    3.97192416e-03,
    1.23238286e-02,
    2.01501112e-03,
    1.11708936e-04,
    -5.26401468e-01,
    1.42209905e-04,
    1.89857613e-04,
    3.44464298e-02,
    2.83970151e-03,
    1.84984813e-04,
    6.80646822e-03,
    1.51128048e-04,
    5.24423579e-05,
    1.42126139e-01,
    1.22098112e-04,
    1.26006711e-01,
    2.10580900e-02,
    4.61128532e-02,
    1.14006475e-01,
    1.37334072e-02,
    1.76880575e-03,
    1.19884737e-01,
    1.57346754e-02,
    1.11708936e-04,
    5.32583238e-05,
    -7.22442182e-01,
    9.72146922e-02,
    2.37529213e-03,
    8.34553326e-02,
    1.84984813e-04,
    7.83858090e-02,
    1.97898249e-01,
    3.18509710e-03,
    7.64271058e-05,
    1.09207238e-01,
    3.12284879e-03,
    6.93207295e-03,
    4.66878693e-04,
    1.70021768e-03,
    6.83856829e-04,
    4.75428648e-02,
    2.62515863e-03,
    1.85578244e-03,
    2.36536969e-01,
    5.32583238e-05,
    7.28171599e-02,
    -6.99008899e-01,
    1.50686418e-01,
    1.41634361e-03,
    2.25340730e-04,
    2.62237077e-04,
    1.39485748e-01,
    1.84375793e-03,
    7.64271058e-05,
    2.93587353e-04,
    1.34245634e-04,
    8.24640894e-03,
    8.75747848e-05,
    1.19226123e-03,
    3.36273557e-02,
    6.85328462e-02,
    1.28011951e-01,
    7.57175006e-03,
    6.34073325e-02,
    9.56382089e-03,
    1.76094828e-03,
    1.49142648e-01,
    -6.29577577e-01,
    5.91424591e-05,
    1.84984813e-04,
    2.62237077e-04,
    1.57688517e-02,
    5.24423579e-05,
    4.45663268e-03,
    1.22098112e-04,
    1.17610184e-01,
    1.05185298e-02,
    2.13111487e-02,
    7.52089070e-05,
    3.11648198e-02,
    2.72035307e-03,
    3.16764801e-01,
    1.79011398e-01,
    4.28269718e-02,
    2.55717711e-03,
    2.00670975e-01,
    4.54671012e-03,
    1.91822822e-04,
    -1.21989108e+00,
    3.81479381e-03,
    9.13481877e-02,
    1.51128048e-04,
    5.24423579e-05,
    2.22491533e-03,
    5.99057294e-02,
    1.02764227e-02,
    2.02505923e-02,
    1.21760303e-03,
    2.81195574e-01,
    1.63482434e-04,
    1.60471410e-04,
    2.37312913e-02,
    1.59339893e-02,
    3.94984924e-04,
    5.32583238e-05,
    1.42209905e-04,
    2.31276570e-04,
    1.91822822e-04,
    1.21964762e-03,
    -1.39491623e+00,
    3.12523134e-01,
    9.74228826e-03,
    1.17539057e-01,
    5.19453638e-02,
    1.00265260e-03,
    3.27030420e-02,
    2.45085742e-01,
    8.75747848e-05,
    2.22871067e-03,
    5.79999708e-01,
    6.90848679e-03,
    2.45770542e-02,
    1.17064419e-04,
    1.95724780e-04,
    1.38234110e-03,
    4.25082472e-02,
    1.89857613e-04,
    1.91822822e-04,
    2.06018025e-02,
    2.20457130e-01,
    -6.43493252e-01,
    2.46337207e-03,
    5.57945783e-02,
    1.30305770e-01,
    5.07334750e-02,
    2.49067536e-02,
    1.24553291e-03,
    1.31159529e-02,
    1.68446893e-03,
    4.61138170e-02,
    1.60471410e-04,
    7.16540366e-01,
    1.85065502e-01,
    1.11708936e-04,
    5.32583238e-05,
    1.86220173e-01,
    1.75231742e-01,
    2.00149851e-02,
    5.91424591e-05,
    1.19248240e-02,
    4.27443812e-03,
    -1.46972686e+00,
    1.34510453e-02,
    5.23767184e-04,
    7.66439155e-04,
    1.35349402e-02,
    1.32939600e-01,
    8.75747848e-05,
    7.52089070e-05,
    8.69167437e-03,
    5.99017309e-04,
    8.80850605e-02,
    1.17064419e-04,
    1.11708936e-04,
    5.32583238e-05,
    8.63714704e-03,
    6.67497599e-03,
    1.91822822e-04,
    5.91424591e-05,
    4.14606461e-01,
    2.78999795e-01,
    3.87631353e-02,
    -1.20633021e+00,
    2.87404131e-03,
    1.22098112e-04,
    1.34245634e-04,
    1.40590268e-01,
    1.56019734e-03,
    7.52089070e-05,
    2.24075563e-01,
    4.53392714e-04,
    1.75193243e-04,
    1.17064419e-04,
    1.11708936e-04,
    9.90407767e-02,
    1.42209905e-04,
    1.89857613e-04,
    1.11856108e-02,
    1.72173161e-03,
    1.25728998e-01,
    4.47105824e-01,
    1.03570469e-03,
    1.97209487e-03,
    -9.84100080e-01,
    1.22098112e-04,
    2.56804380e-02,
    4.02024317e-04,
    1.45259070e-02,
    2.30740927e-01,
    2.36485187e-02,
    6.81095226e-02,
    4.49738575e-02,
    1.73156538e-04,
    7.65559211e-04,
    5.32583238e-05,
    1.27195668e-01,
    4.56516428e-04,
    1.91822822e-04,
    2.90174195e-02,
    1.51906938e-03,
    1.08963177e-01,
    9.48667028e-04,
    5.24423579e-05,
    7.64271058e-05,
    -5.85600233e-01,
    1.44384401e-01,
    5.73193920e-02,
    7.78550346e-04,
    4.57843247e-04,
    1.63482434e-04,
    7.91550908e-02,
    1.21902963e-01,
    3.07755335e-01,
    1.18016022e-02,
    4.99897539e-02,
    3.30811526e-03,
    1.89857613e-04,
    1.68052522e-01,
    4.52731976e-03,
    4.50634103e-02,
    4.86531595e-02,
    1.52370623e-02,
    5.24423579e-05,
    1.46200773e-02,
    1.31319450e-01,
    -1.26842194e+00,
    2.54161020e-01,
    4.35646022e-04,
    9.74743535e-03,
    2.44967964e-03,
    5.10886411e-01,
    1.00514671e-01,
    1.61236805e-01,
    6.13782514e-03,
    7.88190172e-03,
    6.92815372e-03,
    1.10031600e-02,
    1.41801185e-02,
    8.41708380e-03,
    3.18624131e-01,
    2.29548176e-03,
    1.41196517e-01,
    5.18157766e-02,
    2.15935616e-04,
    4.91852710e-02,
    2.39791431e-01,
    -1.65517664e+00,
    8.75747848e-05,
    1.58919429e-03,
    2.31891985e-02,
    1.60471410e-04,
    3.47351439e-02,
    1.17064419e-04,
    1.11708936e-04,
    2.80433834e-02,
    7.58149447e-04,
    1.89857613e-04,
    4.66797001e-02,
    8.22291911e-04,
    1.84984813e-04,
    3.92748798e-02,
    1.51128048e-04,
    9.34292072e-04,
    1.26768570e-02,
    1.08546687e-03,
    6.67812962e-04,
    1.42290353e-04,
    -1.86155755e-01,
    1.92567894e-02,
    1.63482434e-04,
    1.60471410e-04,
    3.47522428e-04,
    4.13043073e-02,
    1.50729537e-02,
    8.07323757e-02,
    3.21488244e-03,
    3.00974819e-03,
    1.91822822e-04,
    2.21125374e-01,
    5.48176595e-03,
    5.87337625e-03,
    1.51128048e-04,
    5.24423579e-05,
    2.34478361e-01,
    7.43286911e-04,
    1.73988786e-02,
    3.00665207e-03,
    2.24229982e-02,
    -6.56110842e-01,
    1.34249486e-03,
    2.44374850e-01,
    2.84672903e-03,
    6.24093832e-04,
    2.34126290e-02,
    4.47398679e-03,
    5.94872566e-04,
    3.90525719e-02,
    3.65673764e-02,
    5.91424591e-05,
    6.56285419e-01,
    7.39697366e-02,
    8.03484357e-03,
    7.18795934e-02,
    1.10555477e-02,
    1.22098112e-04,
    2.01158491e-03,
    2.01832036e-02,
    8.75747848e-05,
    6.17605015e-04,
    -1.19625346e+00,
];

pub(crate) const HIVB_PI_ARR: ProteinFrequencyArray = [
    0.060490222,
    0.066039665,
    0.044127815,
    0.042109048,
    0.020075899,
    0.053606488,
    0.071567447,
    0.072308239,
    0.022293943,
    0.069730629,
    0.098851122,
    0.056968211,
    0.019768318,
    0.028809447,
    0.046025282,
    0.05060433,
    0.053636813,
    0.033011601,
    0.028350243,
    0.061625237,
];
