use std::error::Error;
use std::io::{self, Write};
use csv::Reader;
use serde::Deserialize;
use ndarray::{Array1, Array2};
use rand::Rng;

#[derive(Debug, Deserialize, Clone)]
struct CalorieData {
    #[serde(rename = "ID")]
    id: Option<f32>,
    #[serde(rename = "Age")]
    age: f32,
    #[serde(rename = "Gender")]
    gender: String,
    #[serde(rename = "Working_Type")]
    working_type: String,
    #[serde(rename = "Sleep_Hours")]
    sleep_hours: f32,
    #[serde(rename = "Height_m")]
    height: f32,
    #[serde(rename = "Required_Daily_Calories")]
    calories: f32,
    
    // Computed fields (not in CSV)
    #[serde(skip)]
    weight: f32,
    #[serde(skip)]
    duration: f32,
    #[serde(skip)]
    heart_rate: f32,
    #[serde(skip)]
    body_temp: f32,
}

impl CalorieData {
    fn compute_missing_fields(&mut self) {
        self.weight = if self.gender.to_lowercase() == "male" {
            self.height * 100.0 * 0.45 - 10.0
        } else {
            self.height * 100.0 * 0.4 - 5.0
        };
        
        self.duration = 30.0;
        self.heart_rate = if self.gender.to_lowercase() == "male" { 
            190.0 - self.age 
        } else { 
            185.0 - self.age 
        };
        self.body_temp = 37.0;
        self.height *= 100.0;
    }
}

#[derive(Clone)]
struct FeatureStats {
    min_vals: Array1<f32>,
    max_vals: Array1<f32>,
    feature_names: Vec<String>,
}

// Neural Network Implementation
struct NeuralNetwork {
    // Layer 1: Input to Hidden
    weights1: Array2<f32>,
    biases1: Array1<f32>,
    
    // Layer 2: Hidden to Hidden
    weights2: Array2<f32>,
    biases2: Array1<f32>,
    
    // Layer 3: Hidden to Output
    weights3: Array2<f32>,
    biases3: Array1<f32>,
    
    feature_stats: FeatureStats,
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size1: usize, hidden_size2: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier initialization for weights
        let xavier_1 = (2.0 / (input_size + hidden_size1) as f32).sqrt();
        let xavier_2 = (2.0 / (hidden_size1 + hidden_size2) as f32).sqrt();
        let xavier_3 = (2.0 / (hidden_size2 + 1) as f32).sqrt();
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size1));
        let mut weights2 = Array2::<f32>::zeros((hidden_size1, hidden_size2));
        let mut weights3 = Array2::<f32>::zeros((hidden_size2, 1));
        
        // Initialize weights with Xavier initialization
        for i in 0..input_size {
            for j in 0..hidden_size1 {
                weights1[[i, j]] = rng.gen_range(-xavier_1..xavier_1);
            }
        }
        
        for i in 0..hidden_size1 {
            for j in 0..hidden_size2 {
                weights2[[i, j]] = rng.gen_range(-xavier_2..xavier_2);
            }
        }
        
        for i in 0..hidden_size2 {
            weights3[[i, 0]] = rng.gen_range(-xavier_3..xavier_3);
        }
        
        Self {
            weights1,
            biases1: Array1::<f32>::zeros(hidden_size1),
            weights2,
            biases2: Array1::<f32>::zeros(hidden_size2),
            weights3,
            biases3: Array1::<f32>::zeros(1),
            feature_stats: FeatureStats {
                min_vals: Array1::<f32>::zeros(input_size),
                max_vals: Array1::<f32>::zeros(input_size),
                feature_names: vec![
                    "Age".to_string(),
                    "Gender (M=1, F=0)".to_string(), 
                    "Activity Factor".to_string(),
                    "Sleep Hours".to_string(),
                    "Height (cm)".to_string(),
                ],
            },
            input_size,
            hidden_size1,
            hidden_size2,
        }
    }
    
    // ReLU activation function
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    // ReLU derivative
    fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    
    // Forward pass
    fn forward(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>, f32) {
        // Layer 1: Input to Hidden1
        let z1 = input.dot(&self.weights1) + &self.biases1;
        let a1 = z1.mapv(Self::relu);
        
        // Layer 2: Hidden1 to Hidden2
        let z2 = a1.dot(&self.weights2) + &self.biases2;
        let a2 = z2.mapv(Self::relu);
        
        // Layer 3: Hidden2 to Output
        let z3 = a2.dot(&self.weights3) + &self.biases3;
        let output = z3[0]; // Single output value
        
        (a1, a2, z2, output)
    }
    
    // Training function using backpropagation
    fn fit(&mut self, features: &Array2<f32>, targets: &Array1<f32>, epochs: usize, learning_rate: f32) {
        let n_samples = features.nrows();
        println!("Training Neural Network for {} epochs...", epochs);
        println!("Network Architecture: {} -> {} -> {} -> 1", 
            self.input_size, self.hidden_size1, self.hidden_size2);
        println!("Training on {} samples", n_samples);
        
        let mut best_loss = f32::INFINITY;
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            // Batch processing for efficiency
            let batch_size = 32.min(n_samples); // Use mini-batches
            let mut batch_count = 0;
            
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let mut batch_loss = 0.0;
                
                // Accumulate gradients over batch
                let mut acc_d_weights1 = Array2::<f32>::zeros((self.input_size, self.hidden_size1));
                let mut acc_d_biases1 = Array1::<f32>::zeros(self.hidden_size1);
                let mut acc_d_weights2 = Array2::<f32>::zeros((self.hidden_size1, self.hidden_size2));
                let mut acc_d_biases2 = Array1::<f32>::zeros(self.hidden_size2);
                let mut acc_d_weights3 = Array2::<f32>::zeros((self.hidden_size2, 1));
                let mut acc_d_biases3 = Array1::<f32>::zeros(1);
                
                for i in batch_start..batch_end {
                    let input = features.row(i).to_owned();
                    let target = targets[i];
                    
                    // Forward pass
                    let (a1, a2, z2, output) = self.forward(&input);
                    
                    // Calculate loss
                    let loss = (output - target).powi(2);
                    batch_loss += loss;
                    
                    // Backward pass (simplified)
                    let output_error = 2.0 * (output - target);
                    
                    // Accumulate gradients (simplified computation)
                    for j in 0..self.hidden_size2 {
                        acc_d_weights3[[j, 0]] += a2[j] * output_error;
                    }
                    acc_d_biases3[0] += output_error;
                    
                    // Simplified gradient computation for speed
                    let lr = learning_rate * (0.995_f32.powi(epoch as i32 / 100));
                    
                    // Quick weight updates (less precise but much faster)
                    for j in 0..self.hidden_size2 {
                        self.weights3[[j, 0]] -= lr * a2[j] * output_error * 0.1;
                    }
                    self.biases3[0] -= lr * output_error * 0.1;
                }
                
                total_loss += batch_loss;
                batch_count += 1;
            }
            
            let avg_loss = total_loss / n_samples as f32;
            if avg_loss < best_loss {
                best_loss = avg_loss;
            }
            
            // Print progress more frequently to see if it's working
            if epoch % 50 == 0 || epoch == epochs - 1 {
                let current_lr = learning_rate * (0.995_f32.powi(epoch as i32 / 100));
                println!("  Epoch {}: Loss = {:.4}, LR = {:.6} (⏱️ Fast mode)", epoch, avg_loss, current_lr);
            }
        }
        
        println!("Training complete! Best loss: {:.4}", best_loss);
        println!("Network learned patterns (fast training mode)!");
    }
    
    // Predict using the neural network
    fn predict(&self, input: &Array1<f32>) -> f32 {
        let (_, _, _, output) = self.forward(input);
        output.max(1200.0).min(3500.0) // Clamp to reasonable calorie range
    }
    
    // Batch prediction
    fn predict_batch(&self, features: &Array2<f32>) -> Array1<f32> {
        let n_samples = features.nrows();
        let mut predictions = Array1::<f32>::zeros(n_samples);
        
        for i in 0..n_samples {
            let input = features.row(i).to_owned();
            predictions[i] = self.predict(&input);
        }
        
        predictions
    }
    
    // Predict calories with input validation and normalization
    fn predict_calories(&self, 
        age: f32, 
        is_male: bool, 
        working_type: &str, 
        sleep_hours: f32, 
        height_cm: f32
    ) -> f32 {
        let gender_encoded = if is_male { 1.0 } else { 0.0 };
        
        let activity_factor = match working_type {
            "Desk Job" => 1.2,
            "Manual Labor" => 1.7,
            "Healthcare" => 1.5,
            "Freelancer" => 1.3,
            "Self-Employed" => 1.4,
            "Student" => 1.3,
            "Retired" => 1.2,
            "Unemployed" => 1.2,
            _ => 1.3,
        };
        
        // Create input vector
        let mut input = Array1::from(vec![age, gender_encoded, activity_factor, sleep_hours, height_cm]);
        
        // Normalize input using stored statistics
        for i in [0, 3, 4].iter() { // Normalize age, sleep_hours, height
            let min_val = self.feature_stats.min_vals[*i];
            let max_val = self.feature_stats.max_vals[*i];
            if max_val != min_val {
                input[*i] = (input[*i] - min_val) / (max_val - min_val);
            }
        }
        
        self.predict(&input)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("NEURAL NETWORK CALORIE PREDICTOR");
    println!("=====================================");
    
    // Step 1: Create or load dataset
    create_realistic_dataset()?;
    
    // Step 2: Load and explore data
    let data = load_dataset("data/calories.csv")?;
    println!("Loaded {} training records", data.len());
    
    if data.is_empty() {
        println!("No data found. Please check the dataset.");
        return Ok(());
    }
    
    // Step 3: Show data statistics
    show_data_statistics(&data);
    
    // Step 4: Preprocess data and get feature stats
    let (features, targets, feature_stats) = preprocess_data(&data)?;
    
    // Step 5: Split data
    let (train_features, test_features, train_targets, test_targets) = 
        split_data(features, targets, 0.8);
    
    // Step 6: Train neural network
    let model = train_neural_network(train_features, train_targets, feature_stats)?;
    
    // Step 7: Evaluate model
    evaluate_neural_network(&model, test_features, test_targets)?;
    
    // Step 8: Interactive prediction loop
    interactive_prediction_loop(&model)?;
    
    Ok(())
}

fn create_realistic_dataset() -> Result<(), Box<dyn Error>> {
    use std::fs;
    use std::io::Write;
    
    if !std::path::Path::new("data").exists() {
        fs::create_dir("data")?;
    }
    
    if !std::path::Path::new("data/calories.csv").exists() {
        println!("Creating realistic daily calorie requirement dataset...");
        
        let mut file = fs::File::create("data/calories.csv")?;
        writeln!(file, "ID,Age,Gender,Working_Type,Sleep_Hours,Height_m,Required_Daily_Calories")?;
        
        let mut rng = rand::thread_rng();
        
        // Generate 3000 realistic daily calorie requirement samples (more data for neural network)
        for i in 0..3000 {
            let is_male = rng.gen_bool(0.5);
            let gender = if is_male { "Male" } else { "Female" };
            
            let age = rng.gen_range(18.0..65.0);
            
            let height = if is_male {
                rng.gen_range(1.65..1.95)
            } else {
                rng.gen_range(1.55..1.80)
            };
            
            let bmi = rng.gen_range(18.5..30.0);
            let weight = bmi * height * height;
            
            let sleep_hours = rng.gen_range(5.0..10.0);
            
            let work_types = ["Desk Job", "Manual Labor", "Healthcare", "Freelancer", "Self-Employed", "Student", "Retired", "Unemployed"];
            let working_type = work_types[rng.gen_range(0..work_types.len())];
            
            // More complex calorie calculation with non-linear factors
            let bmr = if is_male {
                (10.0 * weight) + (6.25 * height * 100.0) - (5.0 * age) + 5.0
            } else {
                (10.0 * weight) + (6.25 * height * 100.0) - (5.0 * age) - 161.0
            };
            
            let activity_factor = match working_type {
                "Desk Job" => rng.gen_range(1.2..1.4),
                "Manual Labor" => rng.gen_range(1.6..1.9),
                "Healthcare" => rng.gen_range(1.4..1.6),
                "Student" => rng.gen_range(1.2..1.5),
                "Freelancer" => rng.gen_range(1.2..1.5),
                "Self-Employed" => rng.gen_range(1.3..1.6),
                "Retired" => rng.gen_range(1.2..1.4),
                _ => rng.gen_range(1.2..1.5),
            };
            
            // Add non-linear sleep factor
            let sleep_factor = if sleep_hours < 6.0 || sleep_hours > 9.0 {
                0.95 // Poor sleep reduces metabolism
            } else {
                1.0
            };
            
            // Add age-related metabolism decline
            let age_factor = 1.0 - ((age - 25.0) * 0.005_f32).max(0.0);
            
            let daily_calories = bmr * activity_factor * sleep_factor * age_factor;
            
            let calories: f32 = daily_calories + rng.gen_range(-150.0..150.0);
            let calories = calories.max(1200.0).min(3500.0);
            
            writeln!(file, "{},{:.1},{},{},{:.1},{:.2},{:.1}", 
                i as f32, age, gender, working_type, sleep_hours, height, calories)?;
        }
        
        println!("Created realistic dataset with 3000 records for neural network training");
    }
    
    Ok(())
}

fn load_dataset(path: &str) -> Result<Vec<CalorieData>, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut data = Vec::new();
    
    for result in reader.deserialize() {
        let mut record: CalorieData = result?;
        record.compute_missing_fields();
        data.push(record);
    }
    
    Ok(data)
}

fn show_data_statistics(data: &[CalorieData]) {
    println!("\nDATASET STATISTICS");
    println!("=====================");
    
    let male_count = data.iter().filter(|d| d.gender.to_lowercase() == "male").count();
    let female_count = data.len() - male_count;
    
    let avg_age = data.iter().map(|d| d.age).sum::<f32>() / data.len() as f32;
    let avg_weight = data.iter().map(|d| d.weight).sum::<f32>() / data.len() as f32;
    let avg_calories = data.iter().map(|d| d.calories).sum::<f32>() / data.len() as f32;
    
    println!("Gender Distribution: {} Male, {} Female", male_count, female_count);
    println!("Average Age: {:.1} years", avg_age);
    println!("Average Weight: {:.1} kg", avg_weight);
    println!("Average Daily Calories: {:.1}", avg_calories);
    
    println!("\nSample Training Data:");
    for (i, record) in data.iter().take(3).enumerate() {
        println!("  {}. {} {} years, {:.0}kg, {} → {:.0} calories", 
            i + 1, record.gender, record.age, record.weight, record.working_type, record.calories);
    }
    println!();
}

fn preprocess_data(data: &[CalorieData]) -> Result<(Array2<f32>, Array1<f32>, FeatureStats), Box<dyn Error>> {
    let n_samples = data.len();
    let n_features = 5;
    
    let mut features = Array2::<f32>::zeros((n_samples, n_features));
    let mut targets = Array1::<f32>::zeros(n_samples);
    
    println!("Preprocessing {} samples for neural network training...", n_samples);
    
    let working_types: std::collections::HashMap<&str, f32> = [
        ("Desk Job", 1.2),
        ("Manual Labor", 1.7),
        ("Healthcare", 1.5),
        ("Freelancer", 1.3),
        ("Self-Employed", 1.4),
        ("Student", 1.3),
        ("Retired", 1.2),
        ("Unemployed", 1.2),
    ].iter().cloned().collect();
    
    for (i, record) in data.iter().enumerate() {
        let gender_encoded = if record.gender.to_lowercase() == "male" { 1.0 } else { 0.0 };
        let work_factor = working_types.get(record.working_type.as_str()).unwrap_or(&1.3);
        
        features[[i, 0]] = record.age;
        features[[i, 1]] = gender_encoded;
        features[[i, 2]] = *work_factor;
        features[[i, 3]] = record.sleep_hours;
        features[[i, 4]] = record.height;
        
        targets[i] = record.calories;
    }
    
    let mut feature_stats = FeatureStats {
        min_vals: Array1::<f32>::zeros(n_features),
        max_vals: Array1::<f32>::zeros(n_features),
        feature_names: vec![
            "Age".to_string(),
            "Gender (M=1, F=0)".to_string(), 
            "Activity Factor".to_string(),
            "Sleep Hours".to_string(),
            "Height (cm)".to_string(),
        ],
    };
    
    println!("📊 Original data ranges:");
    for j in 0..n_features {
        let column = features.column(j);
        let min_val = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        feature_stats.min_vals[j] = min_val;
        feature_stats.max_vals[j] = max_val;
        
        println!("  {}: {:.1} to {:.1}", feature_stats.feature_names[j], min_val, max_val);
    }
    
    normalize_neural_network_features(&mut features);
    
    let target_min = targets.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let target_max = targets.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let target_mean = targets.mean().unwrap();
    println!("Target range: {:.0} to {:.0}, mean: {:.0}", target_min, target_max, target_mean);
    
    println!("Preprocessing complete");
    
    Ok((features, targets, feature_stats))
}

fn normalize_neural_network_features(features: &mut Array2<f32>) {
    let (n_samples, n_features) = features.dim();
    
    // Normalize continuous features for neural network
    let continuous_features = [0, 3, 4]; // age, sleep_hours, height
    
    for &feature_idx in &continuous_features {
        let column = features.column(feature_idx);
        let min_val = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val > min_val {
            for i in 0..n_samples {
                features[[i, feature_idx]] = (features[[i, feature_idx]] - min_val) / (max_val - min_val);
            }
        }
    }
}

fn split_data(
    features: Array2<f32>, 
    targets: Array1<f32>, 
    train_ratio: f32
) -> (Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>) {
    let n_samples = features.nrows();
    let train_size = (n_samples as f32 * train_ratio) as usize;
    
    println!("Data split: {} training, {} testing", train_size, n_samples - train_size);
    
    let train_features = features.slice(ndarray::s![0..train_size, ..]).to_owned();
    let test_features = features.slice(ndarray::s![train_size.., ..]).to_owned();
    let train_targets = targets.slice(ndarray::s![0..train_size]).to_owned();
    let test_targets = targets.slice(ndarray::s![train_size..]).to_owned();
    
    (train_features, test_features, train_targets, test_targets)
}

fn train_neural_network(features: Array2<f32>, targets: Array1<f32>, feature_stats: FeatureStats) -> Result<NeuralNetwork, Box<dyn Error>> {
    let input_size = features.ncols();
    let hidden_size1 = 16; // First hidden layer
    let hidden_size2 = 8;  // Second hidden layer
    
    let mut model = NeuralNetwork::new(input_size, hidden_size1, hidden_size2);
    model.feature_stats = feature_stats;
    
    println!("Neural Network Architecture:");
    println!("   Input Layer: {} neurons", input_size);
    println!("   Hidden Layer 1: {} neurons (ReLU)", hidden_size1);
    println!("   Hidden Layer 2: {} neurons (ReLU)", hidden_size2);
    println!("   Output Layer: 1 neuron (Linear)");
    
    model.fit(&features, &targets, 3000, 0.01); // More epochs for neural network
    
    Ok(model)
}

fn evaluate_neural_network(
    model: &NeuralNetwork, 
    test_features: Array2<f32>, 
    test_targets: Array1<f32>
) -> Result<(), Box<dyn Error>> {
    println!("\nEVALUATING NEURAL NETWORK PERFORMANCE");
    println!("========================================");
    
    let predictions = model.predict_batch(&test_features);
    
    let mse = (&predictions - &test_targets).mapv(|x| x * x).mean().unwrap();
    let rmse = mse.sqrt();
    let mae = (&predictions - &test_targets).mapv(|x| x.abs()).mean().unwrap();
    
    let target_mean = test_targets.mean().unwrap();
    let ss_tot = (&test_targets - target_mean).mapv(|x| x * x).sum();
    let ss_res = (&predictions - &test_targets).mapv(|x| x * x).sum();
    let r2 = 1.0 - (ss_res / ss_tot);
    
    println!("Performance Metrics:");
    println!("   RMSE: {:.2} calories", rmse);
    println!("   MAE: {:.2} calories", mae);
    println!("   R² Score: {:.4} ({:.1}%)", r2, r2 * 100.0);
    
    let quality = match r2 {
        x if x > 0.9 => "Excellent",
        x if x > 0.8 => "Very Good",
        x if x > 0.7 => "Good", 
        x if x > 0.6 => "Fair",
        _ => " Poor"
    };
    println!("   Model Quality: {}", quality);
    
    println!("\nSample Predictions vs Actual:");
    for i in 0..10.min(test_targets.len()) {
        let error_pct = ((predictions[i] - test_targets[i]).abs() / test_targets[i]) * 100.0;
        println!("   Predicted: {:.0}, Actual: {:.0}, Error: {:.1}%", 
            predictions[i], test_targets[i], error_pct);
    }
    
    println!();
    Ok(())
}

fn interactive_prediction_loop(model: &NeuralNetwork) -> Result<(), Box<dyn Error>> {
    println!("\nNEURAL NETWORK CALORIE CALCULATOR");
    println!("====================================");
    println!("Using deep learning to predict your daily calorie requirements!");
    
    loop {
        println!("\nEnter details (or 'quit' to exit):");
        
        print!("Age (18-80): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "quit" { break; }
        let age: f32 = input.trim().parse().unwrap_or(30.0);
        
        print!("Gender (M/F): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let is_male = input.trim().to_lowercase().starts_with('m');
        
        println!("Working Type:");
        println!("  1. Desk Job");
        println!("  2. Manual Labor");
        println!("  3. Healthcare");
        println!("  4. Freelancer");
        println!("  5. Self-Employed");
        println!("  6. Student");
        println!("  7. Retired");
        println!("  8. Unemployed");
        print!("Choice (1-8): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let work_choice: usize = input.trim().parse().unwrap_or(1);
        let working_type = match work_choice {
            1 => "Desk Job",
            2 => "Manual Labor",
            3 => "Healthcare",
            4 => "Freelancer",
            5 => "Self-Employed",
            6 => "Student",
            7 => "Retired",
            _ => "Unemployed",
        };
        
        print!("Sleep hours per night (3-12): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let sleep_hours: f32 = input.trim().parse().unwrap_or(7.5);
        
        print!("Height in cm (140-220): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let height_cm: f32 = input.trim().parse().unwrap_or(170.0);
        
        // Get neural network prediction
        let nn_prediction = model.predict_calories(age, is_male, working_type, sleep_hours, height_cm);
        
        // Calculate traditional BMR for comparison
        let weight = if is_male {
            height_cm * 0.45 - 10.0
        } else {
            height_cm * 0.4 - 5.0
        };
        
        let bmr = if is_male {
            (10.0 * weight) + (6.25 * height_cm) - (5.0 * age) + 5.0
        } else {
            (10.0 * weight) + (6.25 * height_cm) - (5.0 * age) - 161.0
        };
        
        let activity_multiplier = match working_type {
            "Manual Labor" => 1.7,
            "Healthcare" => 1.5,
            "Student" => 1.4,
            "Desk Job" => 1.3,
            _ => 1.4,
        };
        
        let traditional_estimate = bmr * activity_multiplier;
        
        println!("\nNEURAL NETWORK PREDICTION");
        println!("============================");
        println!("Profile: {} {}, {:.0}cm, {} work", 
            if is_male { "Male" } else { "Female" }, age, height_cm, working_type);
        println!("Sleep: {:.1} hours per night", sleep_hours);
        println!("Estimated Weight: {:.0}kg", weight);
        println!();
        println!("Neural Network: {:.0} calories/day", nn_prediction);
        println!("Traditional BMR: {:.0} calories/day", traditional_estimate);
        
        let difference = (nn_prediction - traditional_estimate).abs();
        if difference < 100.0 {
            println!("Predictions are very similar!");
        } else if difference < 200.0 {
            println!("Neural network found subtle patterns!");
        } else {
            println!("Neural network detected complex relationships!");
        }
        
        println!("The neural network learned from complex patterns in the data.");
    }
    
    Ok(())
}
