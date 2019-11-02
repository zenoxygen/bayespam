extern crate bayespam;

use bayespam::classifier;

fn main() {
    // Classify a typical spam message
    let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
    let score = classifier::score(spam);
    let is_spam = classifier::is_spam(spam);
    println!("{:.4}", score);
    println!("{}", is_spam);

    // Classifiy a typical ham message
    let ham = "Hi Bob, can you send me your machine learning homework?";
    let score = classifier::score(ham);
    let is_spam = classifier::is_spam(ham);
    println!("{:.4}", score);
    println!("{}", is_spam);
}
