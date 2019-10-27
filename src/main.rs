extern crate bayespam;

use bayespam::classifier;

fn main() {
    // Classify a typical spam message
    let m1 = String::from("Lose up to 19% weight. Special promotion on our new weightloss.");
    let score: f32 = classifier::score(&m1);
    let is_spam: bool = classifier::is_spam(&m1);
    println!("{}", score);
    println!("{}", is_spam);

    // Classifiy a typical ham message
    let m2 = String::from("Hi Bob, can you send me your machine learning homework?");
    let score: f32 = classifier::score(&m2);
    let is_spam: bool = classifier::is_spam(&m2);
    println!("{}", score);
    println!("{}", is_spam);
}
