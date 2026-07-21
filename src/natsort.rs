//! Natural ordering for tensor names: digit runs compare numerically, so
//! `model.layers.2.*` sorts before `model.layers.10.*`. Used for human-facing
//! listings (`--dump-names`); canonical encodings sort bytewise instead.

use std::cmp::Ordering;

pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let si = i;
            while i < a.len() && a[i].is_ascii_digit() {
                i += 1;
            }
            let sj = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let (ra, rb) = (&a[si..i], &b[sj..j]);
            let (ta, tb) = (trim_zeros(ra), trim_zeros(rb));
            // Numeric compare: longer trimmed run is larger; equal lengths
            // compare digitwise; numerically equal runs ("02" vs "2")
            // tie-break bytewise to keep the order total.
            let ord = ta
                .len()
                .cmp(&tb.len())
                .then_with(|| ta.cmp(tb))
                .then_with(|| ra.cmp(rb));
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            let ord = a[i].cmp(&b[j]);
            if ord != Ordering::Equal {
                return ord;
            }
            i += 1;
            j += 1;
        }
    }
    (a.len() - i).cmp(&(b.len() - j))
}

fn trim_zeros(run: &[u8]) -> &[u8] {
    let zeros = run.iter().take_while(|&&c| c == b'0').count();
    if zeros == run.len() {
        &run[run.len() - 1..]
    } else {
        &run[zeros..]
    }
}

#[cfg(test)]
mod tests {
    use super::natural_cmp;
    use std::cmp::Ordering;

    #[test]
    fn numeric_runs() {
        assert_eq!(natural_cmp("layers.2.a", "layers.10.a"), Ordering::Less);
        assert_eq!(natural_cmp("layers.10", "layers.10"), Ordering::Equal);
        assert_eq!(natural_cmp("e.9", "e.11"), Ordering::Less);
        assert_eq!(natural_cmp("a2b3", "a2b10"), Ordering::Less);
    }

    #[test]
    fn total_order_with_leading_zeros() {
        assert_eq!(natural_cmp("a.02", "a.2"), Ordering::Less);
        assert_eq!(natural_cmp("a.2", "a.02"), Ordering::Greater);
        assert_eq!(natural_cmp("a.02", "a.10"), Ordering::Less);
    }

    #[test]
    fn sorts_real_names() {
        let mut v = [
            "model.layers.10.mlp.gate.weight",
            "model.layers.2.mlp.gate.weight",
            "model.embed_tokens.weight",
        ];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v[0], "model.embed_tokens.weight");
        assert_eq!(v[1], "model.layers.2.mlp.gate.weight");
        assert_eq!(v[2], "model.layers.10.mlp.gate.weight");
    }
}
