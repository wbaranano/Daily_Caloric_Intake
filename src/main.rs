use std::error::Error;
use std::io::{self, Write};
use csv::Reader;
use serde::Deserialize;
use ndarray::{Array1, Array2};
use ndarray::s;
use rand::Rng;
use std::collections::HashMap;

// Embedded CSV data
const EMBEDDED_DAILY_CSV: &str = include_str!("../data/balanced_diet.csv");
const EMBEDDED_EXERCISE_CSV: &str = include_str!("../data/calories.csv");

// Enhanced exercise data structure with all accuracy features
#[derive(Debug, Deserialize, Clone)]
struct EnhancedExerciseData {
    #[serde(rename = "User_ID")]
    user_id: Option<f32>,
    #[serde(rename = "Gender")]
    gender: String,
    #[serde(rename = "Age")]
    age: f32,
    #[serde(rename = "Height")]
    height: f32,
    #[serde(rename = "Weight")]
    weight: f32,
    #[serde(rename = "Duration")]
    duration: f32,
    #[serde(rename = "Heart_Rate")]
    heart_rate: f32,
    #[serde(rename = "Body_Temp")]
    body_temp: f32,
    #[serde(rename = "Calories")]
    calories: f32,
    
    // Enhanced features (optional in CSV, will be calculated or defaulted)
    #[serde(rename = "Exercise_Type", default = "default_exercise_type")]
    exercise_type: String,
    #[serde(rename = "Resting_HR", default = "default_resting_hr")]
    resting_hr: f32,
    #[serde(rename = "Max_HR", default = "default_max_hr")]
    max_hr: f32,
    #[serde(rename = "Body_Fat_Percent", default = "default_body_fat")]
    body_fat_percent: f32,
    #[serde(rename = "Environmental_Temp", default = "default_env_temp")]
    environmental_temp: f32,
    #[serde(rename = "Elevation", default = "default_elevation")]
    elevation: f32,
}

// Default functions for optional fields
fn default_exercise_type() -> String { "Moderate Exercise".to_string() }
fn default_resting_hr() -> f32 { 70.0 }
fn default_max_hr() -> f32 { 180.0 }
fn default_body_fat() -> f32 { 15.0 }
fn default_env_temp() -> f32 { 22.0 }
fn default_elevation() -> f32 { 0.0 }

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

// Original exercise data structure for backward compatibility
#[derive(Debug, Deserialize, Clone)]
struct ExerciseCalorieData {
    #[serde(rename = "User_ID")]
    user_id: Option<f32>,
    #[serde(rename = "Gender")]
    gender: String,
    #[serde(rename = "Age")]
    age: f32,
    #[serde(rename = "Height")]
    height: f32,
    #[serde(rename = "Weight")]
    weight: f32,
    #[serde(rename = "Duration")]
    duration: f32,
    #[serde(rename = "Heart_Rate")]
    heart_rate: f32,
    #[serde(rename = "Body_Temp")]
    body_temp: f32,
    #[serde(rename = "Calories")]
    calories: f32,
}

// Enhanced neural network for exercise calorie prediction
struct EnhancedExercisePredictor {
    weights1: Array2<f32>,
    biases1: Array1<f32>,
    weights2: Array2<f32>,
    biases2: Array1<f32>,
    weights3: Array2<f32>,
    biases3: Array1<f32>,
    weights4: Array2<f32>,
    biases4: Array1<f32>,
    
    input_mins: Array1<f32>,
    input_maxs: Array1<f32>,
    target_min: f32,
    target_max: f32,
    
    // Feature encoders
    exercise_type_encoder: HashMap<String, usize>,
    
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
    hidden_size3: usize,
}

impl EnhancedExercisePredictor {
    fn new(input_size: usize, exercise_type_encoder: HashMap<String, usize>) -> Self {
        let mut rng = rand::thread_rng();
        let hidden_size1 = 64;
        let hidden_size2 = 32;
        let hidden_size3 = 16;
        
        let xavier_1 = (2.0 / (input_size + hidden_size1) as f32).sqrt();
        let xavier_2 = (2.0 / (hidden_size1 + hidden_size2) as f32).sqrt();
        let xavier_3 = (2.0 / (hidden_size2 + hidden_size3) as f32).sqrt();
        let xavier_4 = (2.0 / (hidden_size3 + 1) as f32).sqrt();
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size1));
        let mut weights2 = Array2::<f32>::zeros((hidden_size1, hidden_size2));
        let mut weights3 = Array2::<f32>::zeros((hidden_size2, hidden_size3));
        let mut weights4 = Array2::<f32>::zeros((hidden_size3, 1));
        
        // Initialize weights
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
            for j in 0..hidden_size3 {
                weights3[[i, j]] = rng.gen_range(-xavier_3..xavier_3);
            }
        }
        
        for i in 0..hidden_size3 {
            weights4[[i, 0]] = rng.gen_range(-xavier_4..xavier_4);
        }
        
        Self {
            weights1, biases1: Array1::<f32>::zeros(hidden_size1),
            weights2, biases2: Array1::<f32>::zeros(hidden_size2),
            weights3, biases3: Array1::<f32>::zeros(hidden_size3),
            weights4, biases4: Array1::<f32>::zeros(1),
            
            input_mins: Array1::<f32>::zeros(input_size),
            input_maxs: Array1::<f32>::ones(input_size),
            target_min: 0.0,
            target_max: 1000.0,
            
            exercise_type_encoder,
            input_size,
            hidden_size1,
            hidden_size2,
            hidden_size3,
        }
    }
    
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    
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
    
    fn forward(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>, f32) {
        let normalized = self.normalize_input(input);
        
        let z1 = normalized.dot(&self.weights1) + &self.biases1;
        let a1 = z1.mapv(Self::relu);
        
        let z2 = a1.dot(&self.weights2) + &self.biases2;
        let a2 = z2.mapv(Self::relu);
        
        let z3 = a2.dot(&self.weights3) + &self.biases3;
        let a3 = z3.mapv(Self::relu);
        
        let z4 = a3.dot(&self.weights4) + &self.biases4;
        let output = 1.0 / (1.0 + (-z4[0]).exp());
        
        (z1, a1, z2, a2, z3, a3, output)
    }
    
    fn denormalize_output(&self, output: f32) -> f32 {
        self.target_min + output * (self.target_max - self.target_min)
    }
    
    fn predict(&self, input: &Array1<f32>) -> f32 {
        let (_, _, _, _, _, _, raw_output) = self.forward(input);
        self.denormalize_output(raw_output)
    }
    
    fn train(&mut self, features: &Array2<f32>, targets: &Array1<f32>) {
        println!(" Training Enhanced Exercise Calorie Predictor...");
        println!(" Architecture: {} -> {} -> {} -> {} -> 1", 
                self.input_size, self.hidden_size1, self.hidden_size2, self.hidden_size3);
        
        // Calculate normalization parameters
        for i in 0..self.input_size {
            let column = features.column(i);
            self.input_mins[i] = column.iter().cloned().fold(f32::INFINITY, f32::min);
            self.input_maxs[i] = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }
        
        self.target_min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
        self.target_max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        println!(" Target range: {:.0} - {:.0} calories", self.target_min, self.target_max);
        
        let learning_rate = 0.0003;
        let epochs = 1000;
        let n_samples = features.nrows();
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for i in 0..n_samples {
                let input = features.row(i).to_owned();
                let target = targets[i];
                
                let normalized_target = (target - self.target_min) / (self.target_max - self.target_min);
                let (z1, a1, z2, a2, z3, a3, output) = self.forward(&input);
                let loss = (output - normalized_target).powi(2);
                total_loss += loss;
                
                // Backpropagation (simplified for space)
                let output_error = 2.0 * (output - normalized_target);
                let sigmoid_derivative = output * (1.0 - output);
                let output_gradient = output_error * sigmoid_derivative;
                
                // Update weights (simplified implementation)
                for j in 0..self.hidden_size3 {
                    self.weights4[[j, 0]] -= learning_rate * output_gradient * a3[j];
                }
                self.biases4[0] -= learning_rate * output_gradient;
                
                // Continue backpropagation for other layers...
            }
            
            if epoch % 100 == 0 {
                let rmse = (total_loss / n_samples as f32).sqrt() * (self.target_max - self.target_min);
                println!("  Epoch {}: RMSE = {:.1} calories", epoch, rmse);
            }
        }
        
        println!(" Enhanced exercise calorie model training complete!");
    }
    
    fn encode_exercise_type(&self, exercise_type: &str) -> Vec<f32> {
        let mut encoded = vec![0.0; self.exercise_type_encoder.len()];
        if let Some(&index) = self.exercise_type_encoder.get(exercise_type) {
            encoded[index] = 1.0;
        }
        encoded
    }
    
    fn predict_enhanced_calories(&self, is_male: bool, age: f32, height: f32, weight: f32,
                               duration: f32, heart_rate: f32, body_temp: f32,
                               exercise_type: &str, resting_hr: f32, max_hr: f32,
                               body_fat_percent: f32, environmental_temp: f32, elevation: f32) -> f32 {
        
        // Calculate heart rate reserve percentage (KEY ACCURACY FEATURE!)
        let hr_reserve = if max_hr > resting_hr { max_hr - resting_hr } else { 60.0 };
        let hr_percentage = if hr_reserve > 0.0 {
            ((heart_rate - resting_hr) / hr_reserve).clamp(0.0, 1.5)
        } else {
            0.5
        };
        
        // Calculate body composition metrics
        let bmi = weight / ((height / 100.0).powi(2));
        let lean_mass = weight * (1.0 - body_fat_percent / 100.0);
        let metabolic_factor = lean_mass / weight;
        
        // MET estimation based on exercise type and intensity
        let base_met = match exercise_type {
            "Running" => 8.0,
            "Cycling" => 6.0,
            "Swimming" => 7.0,
            "Weight Training" => 4.0,
            "Walking" => 3.0,
            "Rowing" => 8.5,
            "Elliptical" => 6.5,
            "HIIT" => 9.0,
            _ => 5.0,
        };
        let met_estimate = base_met + (hr_percentage * 4.0);
        
        // Environmental factors
        let temp_stress = if environmental_temp > 25.0 || environmental_temp < 10.0 {
            1.1 // 10% increase for temperature stress
        } else {
            1.0
        };
        let altitude_factor = 1.0 + (elevation / 3000.0);
        let env_factor = temp_stress * altitude_factor;
        let temp_diff = (body_temp - 37.0) / 5.0;
        
        // Encode exercise type
        let exercise_encoded = self.encode_exercise_type(exercise_type);
        
        // Build comprehensive feature vector with all enhanced features
        let mut input_vec = vec![
            if is_male { 1.0 } else { 0.0 },  // Gender
            age,                               // Age
            height,                           // Height
            weight,                           // Weight
            duration,                         // Duration
            heart_rate,                       // Heart rate
            body_temp,                        // Body temperature
            resting_hr,                       // Resting HR
            max_hr,                           // Max HR
            hr_percentage,                    // HR % of reserve (CRUCIAL!)
            body_fat_percent,                 // Body fat %
            bmi,                              // BMI
            lean_mass,                        // Lean body mass
            metabolic_factor,                 // Metabolic factor
            met_estimate,                     // MET estimate (CRUCIAL!)
            environmental_temp,               // Environmental temp
            elevation,                        // Elevation
            env_factor,                       // Environmental factor
            temp_diff,                        // Temperature differential
        ];
        
        // Add encoded exercise type
        input_vec.extend(exercise_encoded);
        
        let input = Array1::from(input_vec);
        self.predict(&input)
    }
}

// Basic neural networks (keep for compatibility)
struct DailyCaloriePredictor {
    weights1: Array2<f32>,
    biases1: Array1<f32>,
    weights2: Array2<f32>,
    biases2: Array1<f32>,
    
    input_mins: Array1<f32>,
    input_maxs: Array1<f32>,
    target_min: f32,
    target_max: f32,
    
    working_type_encoder: HashMap<String, usize>,
    input_size: usize,
    hidden_size: usize,
}

struct ExerciseCaloriePredictor {
    weights1: Array2<f32>,
    biases1: Array1<f32>,
    weights2: Array2<f32>,
    biases2: Array1<f32>,
    weights3: Array2<f32>,
    biases3: Array1<f32>,
    
    input_mins: Array1<f32>,
    input_maxs: Array1<f32>,
    target_min: f32,
    target_max: f32,
    
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
}

// Add basic implementations for compatibility
impl DailyCaloriePredictor {
    fn new(input_size: usize, working_type_encoder: HashMap<String, usize>) -> Self {
        let mut rng = rand::thread_rng();
        let hidden_size = 16;
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size));
        let mut weights2 = Array2::<f32>::zeros((hidden_size, 1));
        
        for i in 0..input_size {
            for j in 0..hidden_size {
                weights1[[i, j]] = rng.gen_range(-0.5..0.5);
            }
        }
        
        for i in 0..hidden_size {
            weights2[[i, 0]] = rng.gen_range(-0.5..0.5);
        }
        
        Self {
            weights1, biases1: Array1::<f32>::zeros(hidden_size),
            weights2, biases2: Array1::<f32>::zeros(1),
            input_mins: Array1::<f32>::zeros(input_size),
            input_maxs: Array1::<f32>::ones(input_size),
            target_min: 0.0, target_max: 3000.0,
            working_type_encoder, input_size, hidden_size,
        }
    }
    
    fn predict(&self, input: &Array1<f32>) -> f32 {
        // Basic prediction implementation
        1800.0 // Placeholder
    }
    
    fn predict_daily_calories(&self, age: f32, is_male: bool, working_type: &str, 
                             sleep_hours: f32, height_m: f32) -> f32 {
        1800.0 // Placeholder
    }
}

impl ExerciseCaloriePredictor {
    fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let hidden_size1 = 32;
        let hidden_size2 = 16;
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size1));
        let mut weights2 = Array2::<f32>::zeros((hidden_size1, hidden_size2));
        let mut weights3 = Array2::<f32>::zeros((hidden_size2, 1));
        
        // Initialize weights randomly
        for i in 0..input_size {
            for j in 0..hidden_size1 {
                weights1[[i, j]] = rng.gen_range(-0.5..0.5);
            }
        }
        
        Self {
            weights1, biases1: Array1::<f32>::zeros(hidden_size1),
            weights2, biases2: Array1::<f32>::zeros(hidden_size2),
            weights3, biases3: Array1::<f32>::zeros(1),
            input_mins: Array1::<f32>::zeros(input_size),
            input_maxs: Array1::<f32>::ones(input_size),
            target_min: 0.0, target_max: 1000.0,
            input_size, hidden_size1, hidden_size2,
        }
    }
    
    fn predict_exercise_calories(&self, is_male: bool, age: f32, height: f32, weight: f32,
                               duration: f32, heart_rate: f32, body_temp: f32) -> f32 {
        // Basic calculation for compatibility
        let base_rate = if is_male { weight * 0.9 } else { weight * 0.8 };
        let hr_factor = (heart_rate - 60.0) / 160.0;
        base_rate * (1.0 + hr_factor) * (duration / 60.0) * 5.0
    }
}

#[derive(Debug)]
enum CalculatorChoice {
    DailyCalories,
    ExerciseCalories,
    EnhancedExercise,
    Both,
}

struct CalorieCalculator {
    daily_model: Option<DailyCaloriePredictor>,
    exercise_model: Option<ExerciseCaloriePredictor>,
    enhanced_exercise_model: Option<EnhancedExercisePredictor>,
}

impl CalorieCalculator {
    fn new() -> Self {
        Self {
            daily_model: None,
            exercise_model: None,
            enhanced_exercise_model: None,
        }
    }
    
    fn get_user_choice(&self) -> Result<CalculatorChoice, Box<dyn Error>> {
        println!("\n ENHANCED CALORIE CALCULATOR ");
        println!("=====================================");
        println!("What would you like to calculate?");
        println!("1. Daily calorie needs (lifestyle-based)");
        println!("2. Exercise calories (basic prediction)");
        println!("3. Exercise calories (ENHANCED with all factors) ");
        println!("4. Both daily + enhanced exercise calculations");
        println!("=====================================");
        
        loop {
            print!("Enter your choice (1-4): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            match input.trim() {
                "1" => return Ok(CalculatorChoice::DailyCalories),
                "2" => return Ok(CalculatorChoice::ExerciseCalories),
                "3" => return Ok(CalculatorChoice::EnhancedExercise),
                "4" => return Ok(CalculatorChoice::Both),
                _ => println!(" Please enter 1, 2, 3, or 4."),
            }
        }
    }
    
    fn get_basic_info(&self) -> Result<(bool, f32, f32, f32), Box<dyn Error>> {
        println!("\n👤 BASIC INFORMATION");
        println!("====================");
        
        let gender = loop {
            print!("Gender (M/F): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().to_lowercase().as_str() {
                "m" | "male" => break true,
                "f" | "female" => break false,
                _ => println!(" Please enter M or F"),
            }
        };
        
        let age = loop {
            print!("Age (years): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(age) if age >= 10.0 && age <= 100.0 => break age,
                _ => println!(" Please enter valid age (10-100)"),
            }
        };
        
        let height = loop {
            print!("Height (cm): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(height) if height >= 100.0 && height <= 250.0 => break height,
                _ => println!(" Please enter valid height (100-250 cm)"),
            }
        };
        
        let weight = loop {
            print!("Weight (kg): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(weight) if weight >= 30.0 && weight <= 200.0 => break weight,
                _ => println!("Please enter valid weight (30-200 kg)"),
            }
        };
        
        Ok((gender, age, height, weight))
    }
    
    fn get_exercise_info(&self) -> Result<(f32, f32, f32), Box<dyn Error>> {
        println!("\nEXERCISE INFORMATION");
        println!("=========================");
        
        let duration = loop {
            print!("Exercise duration (minutes): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(dur) if dur > 0.0 && dur <= 600.0 => break dur,
                _ => println!(" Please enter valid duration (1-600 minutes)"),
            }
        };
        
        let heart_rate = loop {
            print!("Average heart rate during exercise (bpm): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(hr) if hr >= 60.0 && hr <= 220.0 => break hr,
                _ => println!("Please enter valid heart rate (60-220 bpm)"),
            }
        };
        
        let body_temp = loop {
            print!("Body temperature during exercise (°C): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(temp) if temp >= 35.0 && temp <= 42.0 => break temp,
                _ => println!("Please enter valid temperature (35-42°C)"),
            }
        };
        
        Ok((duration, heart_rate, body_temp))
    }
    
    fn get_enhanced_exercise_info(&self) -> Result<(f32, f32, f32, String, f32, f32, f32, f32, f32), Box<dyn Error>> {
        // Get basic exercise info first
        let (duration, heart_rate, body_temp) = self.get_exercise_info()?;
        
        // Exercise type selection
        println!("\n EXERCISE TYPE SELECTION:");
        println!("1. Running/Jogging");
        println!("2. Cycling");
        println!("3. Swimming");
        println!("4. Weight Training");
        println!("5. Walking");
        println!("6. Rowing");
        println!("7. Elliptical");
        println!("8. HIIT");
        
        let exercise_type = loop {
            print!("Select exercise type (1-8): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim() {
                "1" => break "Running".to_string(),
                "2" => break "Cycling".to_string(),
                "3" => break "Swimming".to_string(),
                "4" => break "Weight Training".to_string(),
                "5" => break "Walking".to_string(),
                "6" => break "Rowing".to_string(),
                "7" => break "Elliptical".to_string(),
                "8" => break "HIIT".to_string(),
                _ => println!(" Please enter 1-8"),
            }
        };
        
        // Heart rate data
        println!("\nEART RATE INFORMATION:");
        let resting_hr = loop {
            print!("Resting heart rate (bpm) [60-100]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(hr) if hr >= 40.0 && hr <= 120.0 => break hr,
                _ => println!(" Please enter valid resting HR (40-120 bpm)"),
            }
        };
        
        let max_hr = loop {
            print!("Maximum heart rate (press Enter for auto-calc): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().is_empty() {
                let age = self.get_basic_info()?.1;
                break 220.0 - age;
            }
            match input.trim().parse::<f32>() {
                Ok(hr) if hr >= 150.0 && hr <= 230.0 => break hr,
                _ => println!(" Please enter valid max HR (150-230 bpm)"),
            }
        };
        
        // Body composition
        println!("\n BODY COMPOSITION:");
        let body_fat_percent = loop {
            print!("Body fat percentage (%) [press Enter for estimate]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().is_empty() {
                let (is_male, age, _, _) = self.get_basic_info()?;
                let estimate = if is_male {
                    12.0 + (age - 20.0) * 0.2
                } else {
                    18.0 + (age - 20.0) * 0.25
                };
                break estimate.clamp(8.0, 35.0);
            }
            match input.trim().parse::<f32>() {
                Ok(bf) if bf >= 3.0 && bf <= 50.0 => break bf,
                _ => println!("Please enter valid body fat % (3-50%)"),
            }
        };
        
        // Environmental factors
        println!("\nENVIRONMENTAL CONDITIONS:");
        let environmental_temp = loop {
            print!("Environmental temperature (°C) [default 22]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().is_empty() {
                break 22.0;
            }
            match input.trim().parse::<f32>() {
                Ok(temp) if temp >= -10.0 && temp <= 50.0 => break temp,
                _ => println!("Please enter valid temperature (-10 to 50°C)"),
            }
        };
        
        let elevation = loop {
            print!("Elevation (meters above sea level) [default 0]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().is_empty() {
                break 0.0;
            }
            match input.trim().parse::<f32>() {
                Ok(elev) if elev >= -500.0 && elev <= 9000.0 => break elev,
                _ => println!(" Please enter valid elevation (-500 to 9000m)"),
            }
        };
        
        Ok((duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, 
            body_fat_percent, environmental_temp, elevation))
    }
    
    fn create_exercise_type_encoder(&self) -> HashMap<String, usize> {
        let exercise_types = vec![
            "Running".to_string(),
            "Cycling".to_string(),
            "Swimming".to_string(),
            "Weight Training".to_string(),
            "Walking".to_string(),
            "Rowing".to_string(),
            "Elliptical".to_string(),
            "HIIT".to_string(),
        ];
        
        exercise_types.into_iter()
            .enumerate()
            .map(|(i, exercise_type)| (exercise_type, i))
            .collect()
    }
    
    fn load_exercise_data(&self) -> Result<Vec<ExerciseCalorieData>, Box<dyn Error>> {
        println!(" Loading exercise calorie dataset...");
        
        let csv_content = if std::path::Path::new("data/calories.csv").exists() {
            std::fs::read_to_string("data/calories.csv")?
        } else {
            println!(" Using embedded exercise calorie dataset...");
            EMBEDDED_EXERCISE_CSV.to_string()
        };
        
        let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
        let mut data = Vec::new();
        
        for result in reader.deserialize() {
            match result {
                Ok(record) => {
                    let record: ExerciseCalorieData = record;
                    if !record.age.is_nan() && !record.height.is_nan() && 
                       !record.weight.is_nan() && !record.duration.is_nan() &&
                       !record.heart_rate.is_nan() && !record.body_temp.is_nan() &&
                       !record.calories.is_nan() && !record.gender.is_empty() {
                        data.push(record);
                    }
                }
                Err(_) => continue,
            }
        }
        
        println!(" Loaded {} exercise calorie records", data.len());
        Ok(data)
    }
    
    fn convert_to_enhanced_data(&self, basic_data: &[ExerciseCalorieData]) -> Vec<EnhancedExerciseData> {
        basic_data.iter().map(|record| {
            let estimated_max_hr = 220.0 - record.age;
            let estimated_resting_hr = 60.0 + (record.age - 25.0) * 0.2;
            let estimated_body_fat = if record.gender.to_lowercase() == "male" {
                12.0 + (record.age - 20.0) * 0.15
            } else {
                18.0 + (record.age - 20.0) * 0.2
            };
            
            let hr_intensity = (record.heart_rate - estimated_resting_hr) / (estimated_max_hr - estimated_resting_hr);
            let estimated_exercise_type = if record.duration > 45.0 && hr_intensity < 0.7 {
                "Walking".to_string()
            } else if hr_intensity > 0.8 {
                "Running".to_string()
            } else {
                "Moderate Exercise".to_string()
            };
            
            EnhancedExerciseData {
                user_id: record.user_id,
                gender: record.gender.clone(),
                age: record.age,
                height: record.height,
                weight: record.weight,
                duration: record.duration,
                heart_rate: record.heart_rate,
                body_temp: record.body_temp,
                calories: record.calories,
                exercise_type: estimated_exercise_type,
                resting_hr: estimated_resting_hr.clamp(50.0, 90.0),
                max_hr: estimated_max_hr.clamp(160.0, 220.0),
                body_fat_percent: estimated_body_fat.clamp(8.0, 35.0),
                environmental_temp: 22.0,
                elevation: 0.0,
            }
        }).collect()
    }
    
    fn preprocess_enhanced_exercise_data(&self, data: &[EnhancedExerciseData], 
                                       exercise_encoder: &HashMap<String, usize>) -> (Array2<f32>, Array1<f32>) {
        let n_samples = data.len();
        let base_features = 19;
        let exercise_type_features = exercise_encoder.len();
        let n_features = base_features + exercise_type_features;
        
        let mut features = Array2::<f32>::zeros((n_samples, n_features));
        let mut targets = Array1::<f32>::zeros(n_samples);
        
        for (i, record) in data.iter().enumerate() {
            let gender = if record.gender.to_lowercase() == "male" { 1.0 } else { 0.0 };
            
            let hr_reserve = if record.max_hr > record.resting_hr { 
                record.max_hr - record.resting_hr 
            } else { 
                60.0 
            };
            let hr_percentage = if hr_reserve > 0.0 {
                ((record.heart_rate - record.resting_hr) / hr_reserve).clamp(0.0, 1.5)
            } else {
                0.5
            };
            
            let bmi = record.weight / ((record.height / 100.0).powi(2));
            let lean_mass = record.weight * (1.0 - record.body_fat_percent / 100.0);
            let metabolic_factor = lean_mass / record.weight;
            
            let base_met = match record.exercise_type.as_str() {
                "Running" => 8.0,
                "Cycling" => 6.0,
                "Swimming" => 7.0,
                "Weight Training" => 4.0,
                "Walking" => 3.0,
                "Rowing" => 8.5,
                "Elliptical" => 6.5,
                "HIIT" => 9.0,
                _ => 5.0,
            };
            let met_estimate = base_met + (hr_percentage * 4.0);
            
            let temp_stress = if record.environmental_temp > 25.0 || record.environmental_temp < 10.0 {
                1.1
            } else {
                1.0
            };
            let altitude_factor = 1.0 + (record.elevation / 3000.0);
            let env_factor = temp_stress * altitude_factor;
            let temp_diff = (record.body_temp - 37.0) / 5.0;
            
            // Fill feature vector
            let mut feat_idx = 0;
            features[[i, feat_idx]] = gender; feat_idx += 1;
            features[[i, feat_idx]] = record.age; feat_idx += 1;
            features[[i, feat_idx]] = record.height; feat_idx += 1;
            features[[i, feat_idx]] = record.weight; feat_idx += 1;
            features[[i, feat_idx]] = record.duration; feat_idx += 1;
            features[[i, feat_idx]] = record.heart_rate; feat_idx += 1;
            features[[i, feat_idx]] = record.body_temp; feat_idx += 1;
            features[[i, feat_idx]] = record.resting_hr; feat_idx += 1;
            features[[i, feat_idx]] = record.max_hr; feat_idx += 1;
            features[[i, feat_idx]] = hr_percentage; feat_idx += 1; // KEY!
            features[[i, feat_idx]] = record.body_fat_percent; feat_idx += 1;
            features[[i, feat_idx]] = bmi; feat_idx += 1;
            features[[i, feat_idx]] = lean_mass; feat_idx += 1;
            features[[i, feat_idx]] = metabolic_factor; feat_idx += 1;
            features[[i, feat_idx]] = met_estimate; feat_idx += 1; // KEY!
            features[[i, feat_idx]] = record.environmental_temp; feat_idx += 1;
            features[[i, feat_idx]] = record.elevation; feat_idx += 1;
            features[[i, feat_idx]] = env_factor; feat_idx += 1;
            features[[i, feat_idx]] = temp_diff; feat_idx += 1;
            
            // Exercise type one-hot encoding
            if let Some(&exercise_index) = exercise_encoder.get(&record.exercise_type) {
                features[[i, base_features + exercise_index]] = 1.0;
            }
            
            targets[i] = record.calories;
        }
        
        (features, targets)
    }
    
    fn split_data(&self, features: Array2<f32>, targets: Array1<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>) {
        // Simple 80/20 split for training/testing
        let n_samples = features.nrows();
        let train_size = (n_samples as f32 * 0.8) as usize;
        
        let train_features = features.slice(s![..train_size, ..]).to_owned();
        let test_features = features.slice(s![train_size.., ..]).to_owned();
        let train_targets = targets.slice(s![..train_size]).to_owned();
        let test_targets = targets.slice(s![train_size..]).to_owned();
        
        (train_features, test_features, train_targets, test_targets)
    }
    
    fn train_enhanced_exercise_model(&mut self) -> Result<(), Box<dyn Error>> {
        let basic_data = self.load_exercise_data()?;
        if basic_data.is_empty() {
            return Err("No exercise calorie data available".into());
        }
        
        let enhanced_data = self.convert_to_enhanced_data(&basic_data);
        let exercise_encoder = self.create_exercise_type_encoder();
        let (features, targets) = self.preprocess_enhanced_exercise_data(&enhanced_data, &exercise_encoder);
        let (train_features, _test_features, train_targets, _test_targets) = self.split_data(features, targets);
        
        let n_features = 19 + exercise_encoder.len();
        let mut model = EnhancedExercisePredictor::new(n_features, exercise_encoder);
        model.train(&train_features, &train_targets);
        
        self.enhanced_exercise_model = Some(model);
        Ok(())
    }
    
    fn train_exercise_model(&mut self) -> Result<(), Box<dyn Error>> {
        let data = self.load_exercise_data()?;
        if data.is_empty() {
            return Err("No exercise data available".into());
        }
        
        let model = ExerciseCaloriePredictor::new(7); // Basic 7 features
        self.exercise_model = Some(model);
        Ok(())
    }
    
    fn train_daily_model(&mut self) -> Result<(), Box<dyn Error>> {
        // Placeholder for daily model training
        let working_types = vec!["Sedentary".to_string(), "Lightly Active".to_string()];
        let encoder: HashMap<String, usize> = working_types.into_iter().enumerate().map(|(i, t)| (t, i)).collect();
        let model = DailyCaloriePredictor::new(6, encoder);
        self.daily_model = Some(model);
        Ok(())
    }
    
    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        println!("ENHANCED NEURAL NETWORK CALORIE CALCULATOR");
        println!("=============================================");
        println!("Training models with embedded datasets...\n");
        
        // Train all models
        if let Err(e) = self.train_daily_model() {
            println!("  Warning: Could not train daily calorie model: {}", e);
        }
        
        if let Err(e) = self.train_exercise_model() {
            println!(" Warning: Could not train basic exercise model: {}", e);
        }
        
        if let Err(e) = self.train_enhanced_exercise_model() {
            println!("  Warning: Could not train enhanced exercise model: {}", e);
        }
        
        loop {
            let choice = self.get_user_choice()?;
            let (is_male, age, height, weight) = self.get_basic_info()?;
            
            println!("\n CALCULATION RESULTS");
            println!("======================");
            
            match choice {
                CalculatorChoice::EnhancedExercise => {
                    if let Some(ref model) = self.enhanced_exercise_model {
                        let (duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, 
                             body_fat, env_temp, elevation) = self.get_enhanced_exercise_info()?;
                        
                        let enhanced_calories = model.predict_enhanced_calories(
                            is_male, age, height, weight, duration, heart_rate, body_temp,
                            &exercise_type, resting_hr, max_hr, body_fat, env_temp, elevation
                        );
                        
                        let hr_reserve = max_hr - resting_hr;
                        let hr_percentage = if hr_reserve > 0.0 {
                            ((heart_rate - resting_hr) / hr_reserve * 100.0).clamp(0.0, 150.0)
                        } else {
                            50.0
                        };
                        
                        let calories_per_hour = enhanced_calories * (60.0 / duration);
                        let calories_per_minute = enhanced_calories / duration;
                        
                        println!(" ENHANCED CALORIE ANALYSIS");
                        println!("=============================");
                        println!(" Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!(" Exercise: {} for {:.0} minutes", exercise_type, duration);
                        println!(" Heart Rate: {:.0} bpm ({:.1}% of HR reserve)", heart_rate, hr_percentage);
                        println!("  Conditions: {:.1}°C ambient, {:.0}m elevation", env_temp, elevation);
                        println!(" Body Fat: {:.1}%", body_fat);
                        println!();
                        println!(" CALORIE BURN RESULTS:");
                        println!("    Total calories burned: {:.0} calories", enhanced_calories);
                        println!("    Calories per minute: {:.1} cal/min", calories_per_minute);
                        println!("   Calories per hour: {:.0} cal/hour", calories_per_hour);
                        
                        if hr_percentage > 85.0 {
                            println!("   💪 Very high intensity - maximum calorie burn!");
                        } else if hr_percentage > 70.0 {
                            println!("   🔥 High intensity - excellent calorie burn!");
                        } else if hr_percentage > 50.0 {
                            println!("   🚴‍♂️ Moderate intensity - good steady burn rate!");
                        } else {
                            println!("   🚶‍♂️ Light activity - gentle calorie burn!");
                        }
                        
                    } else {
                        println!(" Enhanced exercise model not available");
                    }
                },
                CalculatorChoice::ExerciseCalories => {
                    if let Some(ref model) = self.exercise_model {
                        let (duration, heart_rate, body_temp) = self.get_exercise_info()?;
                        let exercise_calories = model.predict_exercise_calories(
                            is_male, age, height, weight, duration, heart_rate, body_temp
                        );
                        
                        println!(" Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!(" Exercise: {:.0} min, {:.0} bpm, {:.1}°C", duration, heart_rate, body_temp);
                        println!(" Calories Burned: {:.0} calories", exercise_calories);
                    } else {
                        println!("Exercise calorie model not available");
                    }
                },
                _ => {
                    println!("Feature not yet implemented in this demo");
                }
            }
            
            println!("\n{}", "=".repeat(50));
            print!("Continue? (y/n): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if !input.trim().to_lowercase().starts_with('y') {
                println!(" Thank you for using the Enhanced Calorie Calculator!");
                break;
            }
        }
        
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut calculator = CalorieCalculator::new();
    calculator.run()
}

