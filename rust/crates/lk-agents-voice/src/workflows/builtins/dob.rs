use lk_agents_abi::utils::dates::Date;
pub use lk_agents_abi::voice::workflows::GetDOBResult;

pub fn normalize_two_digit_year(year: i64, today: Date) -> i64 {
    if (0..100).contains(&year) {
        let current_yy = i64::from(today.year() % 100);
        year + if year <= current_yy { 2000 } else { 1900 }
    } else {
        year
    }
}
