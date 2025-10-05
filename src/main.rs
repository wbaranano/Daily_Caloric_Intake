use std::error::Error;
use std::io::{self, Write};
use csv::Reader;
use serde::Deserialize;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
struct DailyCalorieData {
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
    height_m: f32,
    #[serde(rename = "Required_Daily_Calories")]
    required_daily_calories: f32,
}

// Neural Network for predicting daily calorie requirements
struct DailyCaloriePredictor {
    // Single hidden layer network
    weights1: Array2<f32>,
    biases1: Array1<f32>,
    weights2: Array2<f32>,
    biases2: Array1<f32>,
    
    // Normalization parameters
    input_mins: Array1<f32>,
    input_maxs: Array1<f32>,
    target_min: f32,
    target_max: f32,
    
    // Encoders for categorical data
    working_type_encoder: HashMap<String, usize>,
    
    input_size: usize,
    hidden_size: usize,
}

impl DailyCaloriePredictor {
    fn new(input_size: usize, working_type_encoder: HashMap<String, usize>) -> Self {
        let mut rng = rand::thread_rng();
        let hidden_size = 16; // Increased for more complex patterns
        
        // Xavier initialization
        let xavier_1 = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let xavier_2 = (2.0 / (hidden_size + 1) as f32).sqrt();
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size));
        let mut weights2 = Array2::<f32>::zeros((hidden_size, 1));
        
        // Initialize weights
        for i in 0..input_size {
            for j in 0..hidden_size {
                weights1[[i, j]] = rng.gen_range(-xavier_1..xavier_1);
            }
        }
        
        for i in 0..hidden_size {
            weights2[[i, 0]] = rng.gen_range(-xavier_2..xavier_2);
        }
        
        Self {
            weights1,
            biases1: Array1::<f32>::zeros(hidden_size),
            weights2,
            biases2: Array1::<f32>::zeros(1),
            
            input_mins: Array1::<f32>::zeros(input_size),
            input_maxs: Array1::<f32>::ones(input_size),
            target_min: 1000.0,
            target_max: 4000.0,
            
            working_type_encoder,
            input_size,
            hidden_size,
        }
    }
    
    // ReLU activation function
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    
    // Normalize inputs to [0, 1] range
    fn normalize_input(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut normalized = Array1::zeros(self.input_size);
        for i in 0..self.input_size {
            if self.input_maxs[i] > self.input_mins[i] {
                normalized[i] = (input[i] - self.input_mins[i]) / (self.input_maxs[i] - self.input_mins[i]);
            } else {
                normalized[i] = 0.5;
            }
        }
        normalized
    }
    
    // Forward pass through the network
    fn forward(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, f32) {
        let normalized = self.normalize_input(input);
        
        // Hidden layer
        let z1 = normalized.dot(&self.weights1) + &self.biases1;
        let a1 = z1.mapv(Self::relu);
        
        // Output layer with sigmoid to keep output in [0,1]
        let z2 = a1.dot(&self.weights2) + &self.biases2;
        let output = 1.0 / (1.0 + (-z2[0]).exp()); // Sigmoid activation
        
        (z1, a1, output)
    }
    
    // Denormalize output back to calorie range
    fn denormalize_output(&self, output: f32) -> f32 {
        self.target_min + output * (self.target_max - self.target_min)
    }
    
    // Make prediction
    fn predict(&self, input: &Array1<f32>) -> f32 {
        let (_, _, raw_output) = self.forward(input);
        self.denormalize_output(raw_output)
    }
    
    // Train the network
    fn train(&mut self, features: &Array2<f32>, targets: &Array1<f32>) {
        println!(" Training Daily Calorie Predictor...");
        println!("Architecture: {} -> {} -> 1", self.input_size, self.hidden_size);
        
        // Calculate normalization parameters
        for i in 0..self.input_size {
            let column = features.column(i);
            self.input_mins[i] = column.iter().cloned().fold(f32::INFINITY, f32::min);
            self.input_maxs[i] = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }
        
        self.target_min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
        self.target_max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        println!(" Target range: {:.0} - {:.0} calories/day", self.target_min, self.target_max);
        
        let learning_rate = 0.001;
        let epochs = 800;
        let n_samples = features.nrows();
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for i in 0..n_samples {
                let input = features.row(i).to_owned();
                let target = targets[i];
                
                // Normalize target to [0,1]
                let normalized_target = (target - self.target_min) / (self.target_max - self.target_min);
                
                // Forward pass
                let (z1, a1, output) = self.forward(&input);
                
                // Calculate loss (MSE)
                let loss = (output - normalized_target).powi(2);
                total_loss += loss;
                
                // Backpropagation
                let output_error = 2.0 * (output - normalized_target);
                
                // Sigmoid derivative
                let sigmoid_derivative = output * (1.0 - output);
                let output_gradient = output_error * sigmoid_derivative;
                
                // Update output layer
                for j in 0..self.hidden_size {
                    self.weights2[[j, 0]] -= learning_rate * output_gradient * a1[j];
                }
                self.biases2[0] -= learning_rate * output_gradient;
                
                // Update hidden layer
                let normalized = self.normalize_input(&input);
                for j in 0..self.hidden_size {
                    let hidden_error = output_gradient * self.weights2[[j, 0]] * Self::relu_derivative(z1[j]);
                    self.biases1[j] -= learning_rate * hidden_error;
                    
                    for k in 0..self.input_size {
                        self.weights1[[k, j]] -= learning_rate * hidden_error * normalized[k];
                    }
                }
            }
            
            if epoch % 100 == 0 {
                let rmse = (total_loss / n_samples as f32).sqrt() * (self.target_max - self.target_min);
                println!("  Epoch {}: RMSE = {:.1} calories", epoch, rmse);
            }
        }
        
        println!(" Training complete!");
    }
    
    // Encode working type to one-hot vector
    fn encode_working_type(&self, working_type: &str) -> Vec<f32> {
        let mut encoded = vec![0.0; self.working_type_encoder.len()];
        if let Some(&index) = self.working_type_encoder.get(working_type) {
            encoded[index] = 1.0;
        }
        encoded
    }
    
    // Predict daily calorie requirements
    fn predict_daily_calories(&self, age: f32, is_male: bool, working_type: &str, 
                             sleep_hours: f32, height_m: f32) -> f32 {
        let gender = if is_male { 1.0 } else { 0.0 };
        let working_encoded = self.encode_working_type(working_type);
        
        let mut input_vec = vec![age, gender, sleep_hours, height_m];
        input_vec.extend(working_encoded);
        
        let input = Array1::from(input_vec);
        self.predict(&input)
    }
}

// Load the balanced_diet.csv data
fn load_calorie_data() -> Result<Vec<DailyCalorieData>, Box<dyn Error>> {
    let file_path = "data/balanced_diet.csv";
    
    if !std::path::Path::new(file_path).exists() {
        return Err(format!("File not found: {}. Please ensure the file exists.", file_path).into());
    }
    
    let mut reader = Reader::from_path(file_path)?;
    let mut data = Vec::new();
    
    for result in reader.deserialize() {
        match result {
            Ok(record) => {
                let record: DailyCalorieData = record;
                // Skip records with missing values
                if !record.age.is_nan() && !record.sleep_hours.is_nan() && 
                   !record.height_m.is_nan() && !record.required_daily_calories.is_nan() &&
                   !record.gender.is_empty() && !record.working_type.is_empty() {
                    data.push(record);
                }
            }
            Err(_) => continue, // Skip malformed records
        }
    }
    
    println!(" Loaded {} valid records from {}", data.len(), file_path);
    Ok(data)
}

// Show dataset statistics
fn show_data_statistics(data: &[DailyCalorieData]) {
    if data.is_empty() {
        return;
    }
    
    let ages: Vec<f32> = data.iter().map(|d| d.age).collect();
    let sleep_hours: Vec<f32> = data.iter().map(|d| d.sleep_hours).collect();
    let heights: Vec<f32> = data.iter().map(|d| d.height_m).collect();
    let calories: Vec<f32> = data.iter().map(|d| d.required_daily_calories).collect();
    
    let avg_age = ages.iter().sum::<f32>() / ages.len() as f32;
    let avg_sleep = sleep_hours.iter().sum::<f32>() / sleep_hours.len() as f32;
    let avg_height = heights.iter().sum::<f32>() / heights.len() as f32;
    let avg_calories = calories.iter().sum::<f32>() / calories.len() as f32;
    
    let min_calories = calories.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_calories = calories.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    println!("\n DATASET STATISTICS");
    println!("=====================");
    println!(" Average age: {:.1} years", avg_age);
    println!(" Average sleep: {:.1} hours", avg_sleep);
    println!(" Average height: {:.2} meters", avg_height);
    println!("  Average daily calories: {:.1}", avg_calories);
    println!(" Calorie range: {:.0} - {:.0}", min_calories, max_calories);
    
    let male_count = data.iter().filter(|d| d.gender.to_lowercase() == "male").count();
    println!(" Males: {}, Females: {}", male_count, data.len() - male_count);
    
    // Show working types
    let mut working_types: HashMap<String, usize> = HashMap::new();
    for record in data {
        *working_types.entry(record.working_type.clone()).or_insert(0) += 1;
    }
    println!("💼 Working types:");
    for (work_type, count) in working_types {
        println!("   {}: {}", work_type, count);
    }
}

// Create working type encoder
fn create_working_type_encoder(data: &[DailyCalorieData]) -> HashMap<String, usize> {
    let mut unique_types: Vec<String> = data.iter()
        .map(|d| d.working_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_types.sort();
    
    unique_types.into_iter()
        .enumerate()
        .map(|(i, work_type)| (work_type, i))
        .collect()
}

// Preprocess data for neural network
fn preprocess_data(data: &[DailyCalorieData], working_type_encoder: &HashMap<String, usize>) -> (Array2<f32>, Array1<f32>) {
    let n_samples = data.len();
    let n_features = 4 + working_type_encoder.len(); // age, gender, sleep, height + one-hot working types
    
    let mut features = Array2::<f32>::zeros((n_samples, n_features));
    let mut targets = Array1::<f32>::zeros(n_samples);
    
    for (i, record) in data.iter().enumerate() {
        let gender = if record.gender.to_lowercase() == "male" { 1.0 } else { 0.0 };
        
        // Basic features
        features[[i, 0]] = record.age;
        features[[i, 1]] = gender;
        features[[i, 2]] = record.sleep_hours;
        features[[i, 3]] = record.height_m;
        
        // One-hot encode working type
        if let Some(&work_index) = working_type_encoder.get(&record.working_type) {
            features[[i, 4 + work_index]] = 1.0;
        }
        
        targets[i] = record.required_daily_calories;
    }
    
    println!(" Preprocessed {} samples with {} features", n_samples, n_features);
    println!(" Features: Age, Gender, Sleep Hours, Height, Working Type (one-hot)");
    (features, targets)
}

// Split data into training and testing sets
fn split_data(features: Array2<f32>, targets: Array1<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>) {
    let n_samples = features.nrows();
    let train_size = (n_samples as f32 * 0.8) as usize;
    
    println!("Training samples: {}, Test samples: {}", train_size, n_samples - train_size);
    
    let train_features = features.slice(ndarray::s![0..train_size, ..]).to_owned();
    let test_features = features.slice(ndarray::s![train_size.., ..]).to_owned();
    let train_targets = targets.slice(ndarray::s![0..train_size]).to_owned();
    let test_targets = targets.slice(ndarray::s![train_size..]).to_owned();
    
    (train_features, test_features, train_targets, test_targets)
}

// Evaluate model performance
fn evaluate_model(model: &DailyCaloriePredictor, test_features: &Array2<f32>, test_targets: &Array1<f32>) {
    println!("\n EVALUATING MODEL PERFORMANCE");
    println!("===============================");
    
    let mut total_error = 0.0;
    let mut total_abs_error = 0.0;
    let n_samples = test_features.nrows();
    
    for i in 0..n_samples {
        let input = test_features.row(i).to_owned();
        let prediction = model.predict(&input);
        let actual = test_targets[i];
        
        let error = prediction - actual;
        total_error += error * error;
        total_abs_error += error.abs();
    }
    
    let rmse = (total_error / n_samples as f32).sqrt();
    let mae = total_abs_error / n_samples as f32;
    
    println!(" RMSE: {:.1} calories", rmse);
    println!(" MAE: {:.1} calories", mae);
    
    // Show a few sample predictions
    println!("\n SAMPLE PREDICTIONS:");
    for i in 0..5.min(n_samples) {
        let input = test_features.row(i).to_owned();
        let prediction = model.predict(&input);
        let actual = test_targets[i];
        let error_pct = ((prediction - actual).abs() / actual) * 100.0;
        println!("   Predicted: {:.0}, Actual: {:.0}, Error: {:.1}%", 
                prediction, actual, error_pct);
    }
    
    println!();
}

// Interactive prediction mode
fn interactive_mode(model: &DailyCaloriePredictor) -> Result<(), Box<dyn Error>> {
    println!(" DAILY CALORIE REQUIREMENT PREDICTOR");
    println!("=======================================");
    println!("Predict your daily calorie needs based on your lifestyle!");
    
    loop {
        println!("\nEnter your details (or 'quit' to exit):");
        
        print!("Age: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim() == "quit" { break; }
        let age: f32 = input.trim().parse().unwrap_or(30.0);
        
        print!("Gender (M/F): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let is_male = input.trim().to_lowercase().starts_with('m');
        
        print!("Working Type (Desk Job/Manual Labor/Healthcare/Freelancer/Student/Retired/Unemployed/Self-Employed): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let working_type = input.trim().to_string();
        
        print!("Sleep hours per night: ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let sleep_hours: f32 = input.trim().parse().unwrap_or(7.5);
        
        print!("Height in meters (e.g., 1.75): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let height_m: f32 = input.trim().parse().unwrap_or(1.70);
        
        // Get prediction
        let predicted_calories = model.predict_daily_calories(
            age, is_male, &working_type, sleep_hours, height_m
        );
        
        println!("\n CALORIE PREDICTION RESULTS");
        println!("=============================");
        println!("Profile: {} {}, {:.1}m tall", 
            if is_male { "Male" } else { "Female" }, age, height_m);
        println!("Lifestyle: {} with {:.1} hours sleep", working_type, sleep_hours);
        println!();
        println!(" Neural Network Prediction: {:.0} calories/day", predicted_calories);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("NEURAL NETWORK DAILY CALORIE PREDICTOR");
    println!("==========================================");
    
    // Load data from balanced_diet.csv
    let data = load_calorie_data()?;
    
    if data.is_empty() {
        println!("No valid data found. Please check that data/balanced_diet.csv exists and contains valid data.");
        return Ok(());
    }
    
    // Show dataset statistics
    show_data_statistics(&data);
    
    // Create working type encoder
    let working_type_encoder = create_working_type_encoder(&data);
    println!("\n🏷️  Working type categories: {:?}", working_type_encoder.keys().collect::<Vec<_>>());
    
    // Preprocess data
    let (features, targets) = preprocess_data(&data, &working_type_encoder);
    let (train_features, test_features, train_targets, test_targets) = split_data(features, targets);
    
    // Train the neural network
    let n_features = 4 + working_type_encoder.len();
    let mut model = DailyCaloriePredictor::new(n_features, working_type_encoder);
    model.train(&train_features, &train_targets);
    
    // Evaluate the model
    evaluate_model(&model, &test_features, &test_targets);
    
    // Interactive prediction mode
    interactive_mode(&model)?;
    
    Ok(())
}
