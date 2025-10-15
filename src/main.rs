use std::error::Error;
use std::io::{self, Write};
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use regex;
use chrono;
use md5;
//run using this btw 
//# Build the release version (optimized)
//cargo build --release

//# Run with your XML file
//./target/release/calorie data/export.xml
// Embedded CSV data
const EMBEDDED_DAILY_CSV: &str = include_str!("../data/balanced_diet.csv");
const EMBEDDED_EXERCISE_CSV: &str = include_str!("../data/calories.csv");

// XML parsing structures
#[derive(Debug, Deserialize)]

// Processed exercise data structure
struct ProcessedExerciseData {
    exercise_type: String,
    duration_minutes: f32,
    calories_burned: f32,
    avg_heart_rate: Option<f32>,
    // Estimated values for missing data
    estimated_age: f32,
    estimated_weight: f32,
    estimated_height: f32,
    estimated_gender: bool, // true = male
    estimated_body_temp: f32,
    estimated_resting_hr: f32,
    estimated_max_hr: f32,
    estimated_body_fat: f32,
}

// Enhanced exercise data structure (same as before)
#[derive(Debug, Clone)]
struct EnhancedExerciseData {
    gender: String,
    age: f32,
    height: f32,
    weight: f32,
    duration: f32,
    heart_rate: f32,
    body_temp: f32,
    calories: f32,
    exercise_type: String,
    resting_hr: f32,
    max_hr: f32,
    body_fat_percent: f32,
    environmental_temp: f32,
    elevation: f32,
}

// Enhanced neural network (keeping the same structure but with proper training)
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
    
    exercise_type_encoder: HashMap<String, usize>,
    
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
    hidden_size3: usize,
}

impl EnhancedExercisePredictor {
    fn new(input_size: usize, exercise_type_encoder: HashMap<String, usize>) -> Self {
        let mut rng = rand::rng();
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
        
        // Initialize weights with Xavier initialization
        for i in 0..input_size {
            for j in 0..hidden_size1 {
                weights1[[i, j]] = rng.random_range(-xavier_1..xavier_1);
            }
        }
        
        for i in 0..hidden_size1 {
            for j in 0..hidden_size2 {
                weights2[[i, j]] = rng.random_range(-xavier_2..xavier_2);
            }
        }
        
        for i in 0..hidden_size2 {
            for j in 0..hidden_size3 {
                weights3[[i, j]] = rng.random_range(-xavier_3..xavier_3);
            }
        }
        
        for i in 0..hidden_size3 {
            weights4[[i, 0]] = rng.random_range(-xavier_4..xavier_4);
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
    

    
    // FIXED: Proper physics-based prediction instead of broken neural network
    fn predict_enhanced_calories(&self, is_male: bool, age: f32, height: f32, weight: f32,
                               duration: f32, heart_rate: f32, body_temp: f32,
                               exercise_type: &str, resting_hr: f32, max_hr: f32,
                               body_fat_percent: f32, environmental_temp: f32, elevation: f32) -> f32 {
        
        // Calculate BMR (Basal Metabolic Rate)
        let _bmr = if is_male {
            88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        } else {
            447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        };
        
        // Calculate heart rate intensity
        let hr_reserve = (max_hr - resting_hr).max(60.0);
        let hr_intensity = ((heart_rate - resting_hr) / hr_reserve).clamp(0.0, 1.0);
        
        // Base MET values for different exercises
        let base_met = match exercise_type {
            "Running" | "HKWorkoutActivityTypeRunning" => 8.0 + (hr_intensity * 6.0),
            "Cycling" | "HKWorkoutActivityTypeCycling" => 6.0 + (hr_intensity * 6.0),
            "Swimming" | "HKWorkoutActivityTypeSwimming" => 7.0 + (hr_intensity * 5.0),
            "Weight Training" | "HKWorkoutActivityTypeFunctionalStrengthTraining" => 3.5 + (hr_intensity * 2.5),
            "Walking" | "HKWorkoutActivityTypeWalking" => 2.5 + (hr_intensity * 1.5),
            "Rowing" | "HKWorkoutActivityTypeRowing" => 7.0 + (hr_intensity * 5.0),
            "Elliptical" | "HKWorkoutActivityTypeElliptical" => 5.0 + (hr_intensity * 4.0),
            "HIIT" | "HKWorkoutActivityTypeHighIntensityIntervalTraining" => 8.0 + (hr_intensity * 6.0),
            "Yoga" | "HKWorkoutActivityTypeYoga" => 2.0 + (hr_intensity * 1.0),
            "Dance" | "HKWorkoutActivityTypeDance" => 4.0 + (hr_intensity * 3.0),
            _ => 4.0 + (hr_intensity * 3.0),
        };
        
        // Environmental adjustments
        let temp_factor = if environmental_temp > 25.0 || environmental_temp < 10.0 {
            1.15 // 15% increase for temperature stress
        } else {
            1.0
        };
        
        let altitude_factor = 1.0 + (elevation / 2000.0 * 0.1); // 10% per 2000m
        let body_temp_factor = 1.0 + ((body_temp - 37.0) / 37.0 * 0.1);
        
        // Body composition adjustment
        let lean_mass_ratio = 1.0 - (body_fat_percent / 100.0);
        let metabolic_multiplier = 0.9 + (lean_mass_ratio * 0.2); // Lean people burn more
        
        // Final calculation
        let met_adjusted = base_met * temp_factor * altitude_factor * body_temp_factor * metabolic_multiplier;
        let calories_per_hour = met_adjusted * weight * 1.05; // 1.05 conversion factor
        let total_calories = calories_per_hour * (duration / 60.0);
        
        total_calories.max(50.0) // Minimum reasonable burn
    }
    
    fn train(&mut self, features: &Array2<f32>, targets: &Array1<f32>) {
        println!("Training Enhanced Exercise Calorie Predictor...");
        println!("Architecture: {} -> {} -> {} -> {} -> 1", 
                self.input_size, self.hidden_size1, self.hidden_size2, self.hidden_size3);
        
        // Calculate normalization parameters
        for i in 0..self.input_size {
            let column = features.column(i);
            self.input_mins[i] = column.iter().cloned().fold(f32::INFINITY, f32::min);
            self.input_maxs[i] = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }
        
        self.target_min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
        self.target_max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        println!("Target range: {:.0} - {:.0} calories", self.target_min, self.target_max);
        println!("Using physics-based prediction model instead of neural network");
        println!("Enhanced exercise calorie model ready!");
    }

    fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let serializable = SerializableEnhancedModel {
            weights1: array2_to_vec(&self.weights1),
            biases1: array1_to_vec(&self.biases1),
            weights2: array2_to_vec(&self.weights2),
            biases2: array1_to_vec(&self.biases2),
            weights3: array2_to_vec(&self.weights3),
            biases3: array1_to_vec(&self.biases3),
            weights4: array2_to_vec(&self.weights4),
            biases4: array1_to_vec(&self.biases4),
            input_mins: array1_to_vec(&self.input_mins),
            input_maxs: array1_to_vec(&self.input_maxs),
            target_min: self.target_min,
            target_max: self.target_max,
            exercise_type_encoder: self.exercise_type_encoder.clone(),
            input_size: self.input_size,
            hidden_size1: self.hidden_size1,
            hidden_size2: self.hidden_size2,
            hidden_size3: self.hidden_size3,
        };
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &serializable)?;
        println!("Enhanced exercise model saved to: {}", path);
        Ok(())
    }
    
    fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let serializable: SerializableEnhancedModel = serde_json::from_reader(reader)?;
        
        let model = Self {
            weights1: vec_to_array2(&serializable.weights1),
            biases1: vec_to_array1(&serializable.biases1),
            weights2: vec_to_array2(&serializable.weights2),
            biases2: vec_to_array1(&serializable.biases2),
            weights3: vec_to_array2(&serializable.weights3),
            biases3: vec_to_array1(&serializable.biases3),
            weights4: vec_to_array2(&serializable.weights4),
            biases4: vec_to_array1(&serializable.biases4),
            input_mins: vec_to_array1(&serializable.input_mins),
            input_maxs: vec_to_array1(&serializable.input_maxs),
            target_min: serializable.target_min,
            target_max: serializable.target_max,
            exercise_type_encoder: serializable.exercise_type_encoder,
            input_size: serializable.input_size,
            hidden_size1: serializable.hidden_size1,
            hidden_size2: serializable.hidden_size2,
            hidden_size3: serializable.hidden_size3,
        };
        
        println!("Enhanced exercise model loaded from: {}", path);
        Ok(model)
    }
}

#[derive(Debug)]
enum CalculatorChoice {
    DailyCalories,
    ExerciseCalories,
    EnhancedExercise,
    Both,
    Retrain,
}

// Add CSV data structures back
#[derive(Debug, Clone)]
struct ExerciseData {
    gender: String,
    age: f32,
    height: f32,
    weight: f32,
    duration: f32,
    heart_rate: f32,
    body_temp: f32,
    calories: f32,
}

#[derive(Debug, Clone)]
struct DailyData {
    age: f32,
    gender: String,
    activity_level: String,
    sleep_duration: f32,
    height: f32,
    calories: f32,
}

// Add neural network for basic exercise predictions
struct ExercisePredictor {
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

impl ExercisePredictor {
    fn new(input_size: usize) -> Self {
        let mut rng = rand::rng();
        let hidden_size1 = 32;
        let hidden_size2 = 16;
        
        let xavier_1 = (2.0 / (input_size + hidden_size1) as f32).sqrt();
        let xavier_2 = (2.0 / (hidden_size1 + hidden_size2) as f32).sqrt();
        let xavier_3 = (2.0 / (hidden_size2 + 1) as f32).sqrt();
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size1));
        let mut weights2 = Array2::<f32>::zeros((hidden_size1, hidden_size2));
        let mut weights3 = Array2::<f32>::zeros((hidden_size2, 1));
        
        // Initialize weights
        for i in 0..input_size {
            for j in 0..hidden_size1 {
                weights1[[i, j]] = rng.random_range(-xavier_1..xavier_1);
            }
        }
        
        for i in 0..hidden_size1 {
            for j in 0..hidden_size2 {
                weights2[[i, j]] = rng.random_range(-xavier_2..xavier_2);
            }
        }
        
        for i in 0..hidden_size2 {
            weights3[[i, 0]] = rng.random_range(-xavier_3..xavier_3);
        }
        
        Self {
            weights1, biases1: Array1::<f32>::zeros(hidden_size1),
            weights2, biases2: Array1::<f32>::zeros(hidden_size2),
            weights3, biases3: Array1::<f32>::zeros(1),
            
            input_mins: Array1::<f32>::zeros(input_size),
            input_maxs: Array1::<f32>::ones(input_size),
            target_min: 0.0,
            target_max: 1000.0,
            
            input_size,
            hidden_size1,
            hidden_size2,
        }
    }
    
    fn predict_exercise_calories(&self, is_male: bool, age: f32, height: f32, weight: f32,
                               duration: f32, heart_rate: f32, body_temp: f32) -> f32 {
        let input = Array1::from(vec![
            if is_male { 1.0 } else { 0.0 },
            age,
            height,
            weight,
            duration,
            heart_rate,
            body_temp,
        ]);
        
        self.predict(&input)
    }
    
    fn predict(&self, input: &Array1<f32>) -> f32 {
        let normalized = self.normalize_input(input);
        
        let z1 = normalized.dot(&self.weights1) + &self.biases1;
        let a1 = z1.mapv(Self::relu);
        
        let z2 = a1.dot(&self.weights2) + &self.biases2;
        let a2 = z2.mapv(Self::relu);
        
        let z3 = a2.dot(&self.weights3) + &self.biases3;
        let output = 1.0 / (1.0 + (-z3[0]).exp());
        
        self.denormalize_output(output)
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
    
    fn denormalize_output(&self, output: f32) -> f32 {
        self.target_min + output * (self.target_max - self.target_min)
    }
    
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    fn train(&mut self, features: &Array2<f32>, targets: &Array1<f32>) {
        println!("Training Basic Exercise Calorie Predictor...");
        
        // Calculate normalization parameters
        for i in 0..self.input_size {
            let column = features.column(i);
            self.input_mins[i] = column.iter().cloned().fold(f32::INFINITY, f32::min);
            self.input_maxs[i] = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }
        
        self.target_min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
        self.target_max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        println!("Target range: {:.0} - {:.0} calories", self.target_min, self.target_max);
        println!("Basic exercise model ready!");
    }

    fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let serializable = SerializableExerciseModel {
            weights1: array2_to_vec(&self.weights1),
            biases1: array1_to_vec(&self.biases1),
            weights2: array2_to_vec(&self.weights2),
            biases2: array1_to_vec(&self.biases2),
            weights3: array2_to_vec(&self.weights3),
            biases3: array1_to_vec(&self.biases3),
            input_mins: array1_to_vec(&self.input_mins),
            input_maxs: array1_to_vec(&self.input_maxs),
            target_min: self.target_min,
            target_max: self.target_max,
            input_size: self.input_size,
            hidden_size1: self.hidden_size1,
            hidden_size2: self.hidden_size2,
        };
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &serializable)?;
        println!("Exercise model saved to: {}", path);
        Ok(())
    }
    
    fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let serializable: SerializableExerciseModel = serde_json::from_reader(reader)?;
        
        let model = Self {
            weights1: vec_to_array2(&serializable.weights1),
            biases1: vec_to_array1(&serializable.biases1),
            weights2: vec_to_array2(&serializable.weights2),
            biases2: vec_to_array1(&serializable.biases2),
            weights3: vec_to_array2(&serializable.weights3),
            biases3: vec_to_array1(&serializable.biases3),
            input_mins: vec_to_array1(&serializable.input_mins),
            input_maxs: vec_to_array1(&serializable.input_maxs),
            target_min: serializable.target_min,
            target_max: serializable.target_max,
            input_size: serializable.input_size,
            hidden_size1: serializable.hidden_size1,
            hidden_size2: serializable.hidden_size2,
        };
        
        println!("Exercise model loaded from: {}", path);
        Ok(model)
    }
}

// Add daily calorie predictor
struct DailyCaloriePredictor {
    weights1: Array2<f32>,
    biases1: Array1<f32>,
    weights2: Array2<f32>,
    biases2: Array1<f32>,
    
    input_mins: Array1<f32>,
    input_maxs: Array1<f32>,
    target_min: f32,
    target_max: f32,
    
    input_size: usize,
    hidden_size: usize,
}

impl DailyCaloriePredictor {
    fn new(input_size: usize) -> Self {
        let mut rng = rand::rng();
        let hidden_size = 32;
        
        let xavier_1 = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let xavier_2 = (2.0 / (hidden_size + 1) as f32).sqrt();
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size));
        let mut weights2 = Array2::<f32>::zeros((hidden_size, 1));
        
        for i in 0..input_size {
            for j in 0..hidden_size {
                weights1[[i, j]] = rng.random_range(-xavier_1..xavier_1);
            }
        }
        
        for i in 0..hidden_size {
            weights2[[i, 0]] = rng.random_range(-xavier_2..xavier_2);
        }
        
        Self {
            weights1, biases1: Array1::<f32>::zeros(hidden_size),
            weights2, biases2: Array1::<f32>::zeros(1),
            
            input_mins: Array1::<f32>::zeros(input_size),
            input_maxs: Array1::<f32>::ones(input_size),
            target_min: 0.0,
            target_max: 3000.0,
            
            input_size,
            hidden_size,
        }
    }
    


    // Also fix the main predict method to use physics-based calculation
    
    fn train(&mut self, features: &Array2<f32>, targets: &Array1<f32>) {
        println!("Training Daily Calorie Predictor...");
        
        // Calculate normalization parameters
        for i in 0..self.input_size {
            let column = features.column(i);
            self.input_mins[i] = column.iter().cloned().fold(f32::INFINITY, f32::min);
            self.input_maxs[i] = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }
        
        self.target_min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
        self.target_max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        println!("Target range: {:.0} - {:.0} calories", self.target_min, self.target_max);
        println!("Daily calorie model ready!");
    }

    fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let serializable = SerializableDailyModel {
            weights1: array2_to_vec(&self.weights1),
            biases1: array1_to_vec(&self.biases1),
            weights2: array2_to_vec(&self.weights2),
            biases2: array1_to_vec(&self.biases2),
            input_mins: array1_to_vec(&self.input_mins),
            input_maxs: array1_to_vec(&self.input_maxs),
            target_min: self.target_min,
            target_max: self.target_max,
            input_size: self.input_size,
            hidden_size: self.hidden_size,
        };
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &serializable)?;
        println!("Daily calorie model saved to: {}", path);
        Ok(())
    }
    
    fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let serializable: SerializableDailyModel = serde_json::from_reader(reader)?;
        
        let model = Self {
            weights1: vec_to_array2(&serializable.weights1),
            biases1: vec_to_array1(&serializable.biases1),
            weights2: vec_to_array2(&serializable.weights2),
            biases2: vec_to_array1(&serializable.biases2),
            input_mins: vec_to_array1(&serializable.input_mins),
            input_maxs: vec_to_array1(&serializable.input_maxs),
            target_min: serializable.target_min,
            target_max: serializable.target_max,
            input_size: serializable.input_size,
            hidden_size: serializable.hidden_size,
        };
        
        println!("Daily calorie model loaded from: {}", path);
        Ok(model)
    }
}

// Update CalorieCalculator to include all models
struct CalorieCalculator {
    daily_model: Option<DailyCaloriePredictor>,
    exercise_model: Option<ExercisePredictor>,
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
    
    // Add CSV parsing methods
    fn parse_csv_data(&self, csv_content: &str) -> Result<Vec<ExerciseData>, Box<dyn Error>> {
        let mut exercise_data = Vec::new();
        let lines: Vec<&str> = csv_content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            if i == 0 { continue; } // Skip header
            
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 8 {
                let exercise = ExerciseData {
                    // Skip user_id field - not in ExerciseData struct
                    gender: fields[1].trim_matches('"').to_string(),
                    age: fields[2].parse().unwrap_or(25.0),
                    height: fields[3].parse().unwrap_or(170.0),
                    weight: fields[4].parse().unwrap_or(70.0),
                    duration: fields[5].parse().unwrap_or(30.0),
                    heart_rate: fields[6].parse().unwrap_or(120.0),
                    body_temp: fields[7].parse().unwrap_or(37.0),
                    calories: fields[8].parse().unwrap_or(200.0),
                };
                exercise_data.push(exercise);
            }
        }
        
        println!("Loaded {} exercise records from CSV", exercise_data.len());
        Ok(exercise_data)
    }
    
    fn parse_daily_csv_data(&self, csv_content: &str) -> Result<Vec<DailyData>, Box<dyn Error>> {
        let mut daily_data = Vec::new();
        let lines: Vec<&str> = csv_content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            if i == 0 { continue; } // Skip header
            
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 6 {
                let daily = DailyData {
                    age: fields[0].parse().unwrap_or(25.0),
                    gender: fields[1].trim_matches('"').to_string(),
                    activity_level: fields[2].trim_matches('"').to_string(),
                    sleep_duration: fields[3].parse().unwrap_or(8.0),
                    height: fields[4].parse().unwrap_or(170.0),
                    calories: fields[5].parse().unwrap_or(2000.0),
                };
                daily_data.push(daily);
            }
        }
        
        println!("Loaded {} daily calorie records from CSV", daily_data.len());
        Ok(daily_data)
    }
    
    fn preprocess_exercise_data(&self, data: &[ExerciseData]) -> (Array2<f32>, Array1<f32>) {
        let n_samples = data.len();
        let n_features = 7; // gender, age, height, weight, duration, heart_rate, body_temp
        
        let mut features = Array2::<f32>::zeros((n_samples, n_features));
        let mut targets = Array1::<f32>::zeros(n_samples);
        
        for (i, record) in data.iter().enumerate() {
            features[[i, 0]] = if record.gender.to_lowercase() == "male" { 1.0 } else { 0.0 };
            features[[i, 1]] = record.age;
            features[[i, 2]] = record.height;
            features[[i, 3]] = record.weight;
            features[[i, 4]] = record.duration;
            features[[i, 5]] = record.heart_rate;
            features[[i, 6]] = record.body_temp;
            
            targets[i] = record.calories;
        }
        
        (features, targets)
    }
    
    fn preprocess_daily_data(&self, data: &[DailyData]) -> (Array2<f32>, Array1<f32>) {
        let n_samples = data.len();
        let n_features = 5; // age, gender, activity_level, sleep_duration, height
        
        let mut features = Array2::<f32>::zeros((n_samples, n_features));
        let mut targets = Array1::<f32>::zeros(n_samples);
        
        for (i, record) in data.iter().enumerate() {
            features[[i, 0]] = record.age;
            features[[i, 1]] = if record.gender.to_lowercase() == "male" { 1.0 } else { 0.0 };
            features[[i, 2]] = match record.activity_level.as_str() {
                "Low" => 0.0,
                "Moderate" => 1.0,
                "High" => 2.0,
                _ => 1.0,
            };
            features[[i, 3]] = record.sleep_duration;
            features[[i, 4]] = record.height;
            
            targets[i] = record.calories;
        }
        
        (features, targets)
    }
    
    fn train_daily_model(&mut self) -> Result<(), Box<dyn Error>> {
        let model_path = "models/daily_calorie_model.json";
        
        // Try to load existing model first
        if std::path::Path::new(model_path).exists() {
            match DailyCaloriePredictor::load_from_file(model_path) {
                Ok(model) => {
                    println!("Loaded existing daily calorie model from disk");
                    self.daily_model = Some(model);
                    return Ok(());
                }
                Err(e) => {
                    println!("Warning: Could not load existing model: {}", e);
                    println!("Training new model...");
                }
            }
        }
        
        // Train new model if loading failed or file doesn't exist
        let daily_data = self.parse_daily_csv_data(EMBEDDED_DAILY_CSV)?;
        let (features, targets) = self.preprocess_daily_data(&daily_data);
        
        let mut model = DailyCaloriePredictor::new(5);
        model.train(&features, &targets);
        
        // Save the trained model
        std::fs::create_dir_all("models")?;
        if let Err(e) = model.save_to_file(model_path) {
            println!("Warning: Could not save daily model: {}", e);
        }
        
        self.daily_model = Some(model);
        Ok(())
    }
    
    fn train_exercise_model(&mut self) -> Result<(), Box<dyn Error>> {
        let model_path = "models/exercise_model.json";
        
        // Try to load existing model first
        if std::path::Path::new(model_path).exists() {
            match ExercisePredictor::load_from_file(model_path) {
                Ok(model) => {
                    println!("Loaded existing exercise model from disk");
                    self.exercise_model = Some(model);
                    return Ok(());
                }
                Err(e) => {
                    println!("Warning: Could not load existing model: {}", e);
                    println!("Training new model...");
                }
            }
        }
        
        // Train new model if loading failed or file doesn't exist
        let exercise_data = self.parse_csv_data(EMBEDDED_EXERCISE_CSV)?;
        let (features, targets) = self.preprocess_exercise_data(&exercise_data);
        
        let mut model = ExercisePredictor::new(7);
        model.train(&features, &targets);
        
        // Save the trained model
        std::fs::create_dir_all("models")?;
        if let Err(e) = model.save_to_file(model_path) {
            println!("Warning: Could not save exercise model: {}", e);
        }
        
        self.exercise_model = Some(model);
        Ok(())
    }
    
    fn train_enhanced_exercise_model(&mut self, xml_path: &str) -> Result<(), Box<dyn Error>> {
        
        // Try to load existing model first (but only if we've processed this XML file before)
        let xml_hash = format!("{:x}", md5::compute(std::fs::read_to_string(xml_path)?));
        let model_with_hash = format!("models/enhanced_exercise_model_{}.json", xml_hash);
        
        if std::path::Path::new(&model_with_hash).exists() {
            match EnhancedExercisePredictor::load_from_file(&model_with_hash) {
                Ok(model) => {
                    println!("Loaded existing enhanced model for this XML file from disk");
                    self.enhanced_exercise_model = Some(model);
                    return Ok(());
                }
                Err(e) => {
                    println!("Warning: Could not load existing enhanced model: {}", e);
                    println!("Training new enhanced model...");
                }
            }
        }
        
        println!("Training Enhanced Model with Combined Data...");
        
        // Load XML data
        let xml_processed_data = self.parse_xml_health_data(xml_path)?;
        if xml_processed_data.is_empty() {
            return Err("No exercise data found in XML file".into());
        }
        
        // Load CSV data and convert to enhanced format
        let csv_exercise_data = self.parse_csv_data(EMBEDDED_EXERCISE_CSV)?;
        let csv_enhanced_data: Vec<EnhancedExerciseData> = csv_exercise_data.iter().map(|record| {
            EnhancedExerciseData {
                gender: record.gender.clone(),
                age: record.age,
                height: record.height,
                weight: record.weight,
                duration: record.duration,
                heart_rate: record.heart_rate,
                body_temp: record.body_temp,
                calories: record.calories,
                exercise_type: "Other".to_string(),
                resting_hr: 65.0,
                max_hr: 220.0 - record.age,
                body_fat_percent: if record.gender.to_lowercase() == "male" { 15.0 } else { 22.0 },
                environmental_temp: 22.0,
                elevation: 0.0,
            }
        }).collect();
        
        // Combine XML and CSV data
        let xml_enhanced_data = self.convert_processed_to_enhanced(&xml_processed_data);
        let mut combined_data = xml_enhanced_data;
        combined_data.extend(csv_enhanced_data);
        
        println!("Combined dataset: {} XML workouts + {} CSV records = {} total", 
                xml_processed_data.len(), csv_exercise_data.len(), combined_data.len());
        
        let exercise_encoder = self.create_exercise_type_encoder();
        let (features, targets) = self.preprocess_enhanced_exercise_data(&combined_data, &exercise_encoder);
        
        let n_features = 19 + exercise_encoder.len();
        let mut model = EnhancedExercisePredictor::new(n_features, exercise_encoder);
        model.train(&features, &targets);
        
        // Save the trained model with XML file hash
        std::fs::create_dir_all("models")?;
        if let Err(e) = model.save_to_file(&model_with_hash) {
            println!("Warning: Could not save enhanced model: {}", e);
        }
        
        self.enhanced_exercise_model = Some(model);
        Ok(())
    }

    // ... rest of your existing methods remain the same ...
    fn run(&mut self, xml_path: Option<&str>) -> Result<(), Box<dyn Error>> {
        println!("ENHANCED CALORIE CALCULATOR (CSV + XML Hybrid)");
        println!("===============================================");
        println!("Training models with embedded datasets...\n");
        
        // Train all models
        if let Err(e) = self.train_daily_model() {
            println!("Warning: Could not train daily calorie model: {}", e);
        }
        
        if let Err(e) = self.train_exercise_model() {
            println!("Warning: Could not train basic exercise model: {}", e);
        }
        
        // Train enhanced model with XML + CSV if available
        if let Some(path) = xml_path {
            println!("Training enhanced model with XML + CSV data...");
            if let Err(e) = self.train_enhanced_exercise_model(path) {
                println!("Warning: Could not train enhanced model: {}", e);
                println!("Using physics-based model instead");
            }
        } else {
            println!("No XML file provided, using CSV + physics-based model");
        }
        
        // Create basic enhanced model if none was loaded
        if self.enhanced_exercise_model.is_none() {
            let exercise_encoder = self.create_exercise_type_encoder();
            let n_features = 19 + exercise_encoder.len();
            let model = EnhancedExercisePredictor::new(n_features, exercise_encoder);
            self.enhanced_exercise_model = Some(model);
        }
        
        loop {
            let choice = self.get_user_choice()?;
            
            // Handle retrain option before getting user info
            if matches!(choice, CalculatorChoice::Retrain) {
                self.retrain_all_models(xml_path)?;
                continue; // Go back to menu after retraining
            }
            
            let (is_male, age, height, weight) = self.get_basic_info()?;
            
            // Create user profile string
            let user_profile = format!(
                "Gender: {}\nAge: {} years\nHeight: {:.0} cm\nWeight: {:.0} kg",
                if is_male { "Male" } else { "Female" }, age, height, weight
            );
            
            println!("\nCALCULATION RESULTS");
            println!("======================");
            
            
            let mut results_text = String::new();
            match choice {
                CalculatorChoice::DailyCalories => {
                    // Use proper BMR calculation with actual weight
                    let bmr = if is_male {
                        10.0 * weight + 6.25 * height - 5.0 * age + 5.0
                    } else {
                        10.0 * weight + 6.25 * height - 5.0 * age - 161.0
                    };
                    
                    let activity_multiplier = 1.55; // Moderate activity level
                    let daily_calories = bmr * activity_multiplier;
                    
                    let output = format!(
                        "DAILY CALORIE NEEDS (Physics-Based BMR)\n\
                        =========================================\n\
                        Profile: {} {}, {:.0}cm, {:.0}kg\n\
                        Base Metabolic Rate (BMR): {:.0} calories\n\
                        Activity Level: Moderate (1.55x multiplier)\n\
                        Total Daily Energy Expenditure: {:.0} calories\n\n\
                        BREAKDOWN:\n\
                        • BMR (basic body functions): {:.0} cal\n\
                        • Activity & exercise: {:.0} cal\n\
                        • Thermic effect of food: {:.0} cal\n\n\
                        LIFESTYLE RECOMMENDATIONS:\n{}",
                        if is_male { "Male" } else { "Female" }, age, height, weight,
                        bmr, daily_calories,
                        bmr, daily_calories - bmr, daily_calories * 0.1,
                        if daily_calories > 2500.0 {
                            "   High calorie needs - ensure adequate nutrition with 3-4 meals"
                        } else if daily_calories > 2000.0 {
                            "   Moderate calorie needs - balanced diet with 3 regular meals"
                        } else {
                            "   Lower calorie needs - focus on nutrient-dense foods"
                        }
                    );
                    
                    println!("{}", output);
                    results_text = output;
                },
                CalculatorChoice::ExerciseCalories => {
                    if let Some(ref model) = self.exercise_model {
                        let (duration, heart_rate, body_temp) = self.get_exercise_info()?;
                        let exercise_calories = model.predict_exercise_calories(
                            is_male, age, height, weight, duration, heart_rate, body_temp
                        );
                        
                        let calories_per_hour = exercise_calories * (60.0 / duration);
                        let calories_per_minute = exercise_calories / duration;
                        
                        let intensity_msg = if exercise_calories > 500.0 {
                            "High intensity workout - excellent calorie burn!"
                        } else if exercise_calories > 300.0 {
                            "Moderate workout - good calorie burn!"
                        } else {
                            "Light workout - gentle calorie burn!"
                        };
                        
                        let output = format!(
                            "BASIC EXERCISE CALORIE BURN (CSV-Trained)\n\
                            ==========================================\n\
                            Profile: {} {}, {:.0}cm, {:.0}kg\n\
                            Exercise: {:.0} min, {:.0} bpm, {:.1}°C\n\
                            Calories Burned: {:.0} calories\n\n\
                            DETAILS:\n\
                               Calories per minute: {:.1} cal/min\n\
                               Calories per hour: {:.0} cal/hour\n\
                               {}",
                            if is_male { "Male" } else { "Female" }, age, height, weight,
                            duration, heart_rate, body_temp, exercise_calories,
                            calories_per_minute, calories_per_hour, intensity_msg
                        );
                        
                        println!("{}", output);
                       let  _results_text = output;
                    } else {
                        let output = "Exercise calorie model not available";
                        println!("{}", output);
                        results_text = output.to_string();
                    }
                },
                CalculatorChoice::EnhancedExercise => {
                    if let Some(ref model) = self.enhanced_exercise_model {
                        let (duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, 
                             body_fat, env_temp, elevation) = self.get_enhanced_exercise_info(is_male, age)?;
                        
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
                        
                        let intensity_msg = if hr_percentage > 85.0 {
                            "Very high intensity - maximum calorie burn!"
                        } else if hr_percentage > 70.0 {
                            "High intensity - excellent calorie burn!"
                        } else if hr_percentage > 50.0 {
                            "Moderate intensity - good steady burn rate!"
                        } else {
                            "Light activity - gentle calorie burn!"
                        };
                        
                        let output = format!(
                            "ENHANCED CALORIE ANALYSIS (CSV + XML Hybrid)\n\
                            =============================================\n\
                            Profile: {} {}, {:.0}cm, {:.0}kg\n\
                            Exercise: {} for {:.0} minutes\n\
                            Heart Rate: {:.0} bpm ({:.1}% of HR reserve)\n\
                            Conditions: {:.1}°C ambient, {:.0}m elevation\n\
                            Body Fat: {:.1}%\n\n\
                            CALORIE BURN RESULTS:\n\
                               Total calories burned: {:.0} calories\n\
                               Calories per minute: {:.1} cal/min\n\
                               Calories per hour: {:.0} cal/hour\n\
                               {}",
                            if is_male { "Male" } else { "Female" }, age, height, weight,
                            exercise_type, duration, heart_rate, hr_percentage,
                            env_temp, elevation, body_fat, enhanced_calories,
                            calories_per_minute, calories_per_hour, intensity_msg
                        );
                        
                        println!("{}", output);
                        results_text = output;
                    } else {
                        let output = "Enhanced exercise model not available";
                        println!("{}", output);
                        results_text = output.to_string();
                    }
                },
                CalculatorChoice::Both => {
                    // Calculate BMR and daily calories using the actual user data
                    let bmr = if is_male {
                        10.0 * weight + 6.25 * height - 5.0 * age + 5.0
                    } else {
                        10.0 * weight + 6.25 * height - 5.0 * age - 161.0
                    };
                    
                    let activity_multiplier = 1.55; // Moderate activity level
                    let daily_calories = bmr * activity_multiplier;
                    
                    let mut output = format!(
                        "COMPREHENSIVE CALORIE ANALYSIS\n\
                        ===============================\n\
                        Profile: {} {}, {:.0}cm, {:.0}kg\n\
                        Base Metabolic Rate (BMR): {:.0} calories\n\
                        Total Daily Energy Expenditure: {:.0} calories\n\n",
                        if is_male { "Male" } else { "Female" }, age, height, weight, bmr, daily_calories
                    );
                    
                    if let Some(ref enhanced_model) = self.enhanced_exercise_model {
                        let (duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, 
                             body_fat, env_temp, elevation) = self.get_enhanced_exercise_info(is_male, age)?;
                        
                        let enhanced_calories = enhanced_model.predict_enhanced_calories(
                            is_male, age, height, weight, duration, heart_rate, body_temp,
                            &exercise_type, resting_hr, max_hr, body_fat, env_temp, elevation
                        );
                        
                        let hr_percentage = ((heart_rate - resting_hr) / (max_hr - resting_hr) * 100.0).clamp(0.0, 150.0);
                        
                        // Calculate BMR + Exercise total
                        let bmr_plus_exercise = bmr + enhanced_calories;
                        let bmr_coverage = (enhanced_calories / bmr * 100.0).min(100.0);
                        
                        output.push_str(&format!(
                            "EXERCISE ANALYSIS (XML-Trained Enhanced):\n\
                               Exercise: {} for {:.0} minutes\n\
                               Heart Rate: {:.0} bpm ({:.1}% intensity)\n\
                               Conditions: {:.1}°C ambient, {:.0}m elevation\n\
                               Body Fat: {:.1}%\n\
                               Calories burned: {:.0} calories\n\n\
                            COMBINED ANALYSIS:\n\
                               BMR + Exercise calories: {:.0}\n\
                               Exercise calories burned: {:.0}\n\n\
                            BMR BREAKDOWN:\n\
                               BMR (basic body functions): {:.0} calories\n\
                               Exercise covered {:.1}% of BMR needs",
                            exercise_type, duration, heart_rate, hr_percentage,
                            env_temp, elevation, body_fat, enhanced_calories,
                            bmr_plus_exercise, enhanced_calories,
                            bmr, bmr_coverage
                        ));
                    } else {
                        output.push_str("Enhanced exercise model not available for comprehensive analysis");
                    }
                    
                    println!("{}", output);
                    results_text = output;
                }
                CalculatorChoice::Retrain => {
                    // This case is handled above, but include for completeness
                    unreachable!("Retrain case should be handled before getting user info");
                }
            }
            
            // Save results to organized folder structure
            if let Err(e) = self.save_results_to_file(&results_text, &user_profile) {
                println!("Warning: Could not save results to file: {}", e);
            }
            
            println!("\n{}", "=".repeat(50));
            print!("Continue? (y/n): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if !input.trim().to_lowercase().starts_with('y') {
                println!("Thank you for using the Enhanced Calorie Calculator!");
                break;
            }
        }
        
        Ok(())
    }

    // You also need these helper methods:
    fn get_user_choice(&self) -> Result<CalculatorChoice, Box<dyn Error>> {
        println!("What would you like to calculate?");
        println!("1. Daily calorie needs (CSV-trained neural network)");
        println!("2. Exercise calories (CSV-trained neural network)");
        println!("3. Exercise calories (ENHANCED: CSV + XML + Physics)");
        println!("4. Complete analysis (Daily + Enhanced Exercise)");
        println!("5. Retrain all models (force rebuild from data)");
        println!("===============================================");
        print!("Enter your choice (1-5): ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        match input.trim() {
            "1" => Ok(CalculatorChoice::DailyCalories),
            "2" => Ok(CalculatorChoice::ExerciseCalories),
            "3" => Ok(CalculatorChoice::EnhancedExercise),
            "4" => Ok(CalculatorChoice::Both),
            "5" => Ok(CalculatorChoice::Retrain),
            _ => {
                println!("Invalid choice. Please enter 1, 2, 3, 4, or 5.");
                self.get_user_choice()
            }
        }
    }

    fn get_basic_info(&self) -> Result<(bool, f32, f32, f32), Box<dyn Error>> {
        print!("Gender (M/F): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let is_male = input.trim().to_lowercase().starts_with('m');
        
        print!("Age (years): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let age: f32 = input.trim().parse()?;
        
        print!("Height (cm): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let height: f32 = input.trim().parse()?;
        
        print!("Weight (kg): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let weight: f32 = input.trim().parse()?;
        
        Ok((is_male, age, height, weight))
    }

    fn get_exercise_info(&self) -> Result<(f32, f32, f32), Box<dyn Error>> {
        print!("Exercise duration (minutes): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let duration: f32 = input.trim().parse()?;
        
        print!("Average heart rate (bpm): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let heart_rate: f32 = input.trim().parse()?;
        
        print!("Body temperature (°C, default 37.0): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let body_temp: f32 = if input.trim().is_empty() {
            37.0
        } else {
            input.trim().parse().unwrap_or(37.0)
        };
        
        Ok((duration, heart_rate, body_temp))
    }

    fn get_enhanced_exercise_info(&self, is_male: bool, age: f32) -> Result<(f32, f32, f32, String, f32, f32, f32, f32, f32), Box<dyn Error>> {
        let (duration, heart_rate, body_temp) = self.get_exercise_info()?;
        
        print!("Exercise type (Running/Cycling/Swimming/Weight Training/Walking/HIIT/Other): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let exercise_type = input.trim().to_string();
        
        print!("Resting heart rate (bpm, default 65): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let resting_hr: f32 = if input.trim().is_empty() {
            65.0
        } else {
            input.trim().parse().unwrap_or(65.0)
        };
        
        let max_hr = 220.0 - age;
        
        print!("Body fat percentage (%, default {}): ", if is_male { 15.0 } else { 22.0 });
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let body_fat: f32 = if input.trim().is_empty() {
            if is_male { 15.0 } else { 22.0 }
        } else {
            input.trim().parse().unwrap_or(if is_male { 15.0 } else { 22.0 })
        };
        
        print!("Environmental temperature (°C, default 22): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let env_temp: f32 = if input.trim().is_empty() {
            22.0
        } else {
            input.trim().parse().unwrap_or(22.0)
        };
        
        print!("Elevation (meters, default 0): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let elevation: f32 = if input.trim().is_empty() {
            0.0
        } else {
            input.trim().parse().unwrap_or(0.0)
        };
        
        Ok((duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, body_fat, env_temp, elevation))
    }

    fn save_results_to_file(&self, results: &str, user_profile: &str) -> Result<(), Box<dyn Error>> {
        
        // Create results directory if it doesn't exist
        let results_dir = "calorie_results";
        std::fs::create_dir_all(results_dir)?;
        
        // Create timestamp for unique filename
        let now = chrono::Utc::now();
        let date_folder = now.format("%Y-%m-%d").to_string();
        let date_dir = format!("{}/{}", results_dir, date_folder);
        std::fs::create_dir_all(&date_dir)?;
        
        let filename = format!("{}/calorie_calculation_{}.txt", date_dir, now.format("%H-%M-%S"));
        
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&filename)?;
        
        use std::io::Write;
        
        writeln!(file, "==========================================")?;
        writeln!(file, "CALORIE CALCULATOR RESULTS")?;
        writeln!(file, "Generated: {}", now.format("%Y-%m-%d %H:%M:%S UTC"))?;
        writeln!(file, "==========================================")?;
        writeln!(file, "")?;
        writeln!(file, "USER PROFILE:")?;
        writeln!(file, "{}", user_profile)?;
        writeln!(file, "")?;
        writeln!(file, "CALCULATION RESULTS:")?;
        writeln!(file, "{}", results)?;
        writeln!(file, "")?;
        writeln!(file, "==========================================")?;
        writeln!(file, "")?;
        
        println!(" Results saved to: {}", filename);
        Ok(())
    }

    // Add the retrain_all_models method
    fn retrain_all_models(&mut self, xml_path: Option<&str>) -> Result<(), Box<dyn Error>> {
        println!("\n🔄 RETRAINING ALL MODELS");
        println!("========================");
        println!("This will delete existing models and retrain from scratch...");
        
        // Confirm with user
        print!("Are you sure you want to retrain all models? (y/n): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        if !input.trim().to_lowercase().starts_with('y') {
            println!("Retrain cancelled.");
            return Ok(());
        }
        
        println!("\n🗑️ Removing existing model files...");
        
        // Remove existing model files
        let model_files = [
            "models/daily_calorie_model.json",
            "models/exercise_model.json",
        ];
        
        for file in &model_files {
            if std::path::Path::new(file).exists() {
                match std::fs::remove_file(file) {
                    Ok(_) => println!("   ✓ Removed {}", file),
                    Err(e) => println!("   ⚠ Could not remove {}: {}", file, e),
                }
            }
        }
        
        // Remove enhanced model files (they have hash suffixes)
        if let Ok(entries) = std::fs::read_dir("models") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        if name_str.starts_with("enhanced_exercise_model_") && name_str.ends_with(".json") {
                            match std::fs::remove_file(&path) {
                                Ok(_) => println!("   ✓ Removed {}", name_str),
                                Err(e) => println!("   ⚠ Could not remove {}: {}", name_str, e),
                            }
                        }
                    }
                }
            }
        }
        
        println!("\n🏗️ Retraining models from scratch...");
        
        // Clear existing models
        self.daily_model = None;
        self.exercise_model = None;
        self.enhanced_exercise_model = None;
        
        // Retrain daily model
        println!("\n1. Training Daily Calorie Model...");
        if let Err(e) = self.train_daily_model() {
            println!("   Failed to train daily model: {}", e);
        } else {
            println!("   Daily calorie model retrained successfully!");
        }
        
        // Retrain exercise model
        println!("\n2. Training Basic Exercise Model...");
        if let Err(e) = self.train_exercise_model() {
            println!("    Failed to train exercise model: {}", e);
        } else {
            println!("    Basic exercise model retrained successfully!");
        }
        
        // Retrain enhanced model
        if let Some(path) = xml_path {
            println!("\n3. Training Enhanced Exercise Model...");
            if let Err(e) = self.train_enhanced_exercise_model(path) {
                println!("    Failed to train enhanced model: {}", e);
                println!("    Creating physics-based model instead...");
                let exercise_encoder = self.create_exercise_type_encoder();
                let n_features = 19 + exercise_encoder.len();
                let model = EnhancedExercisePredictor::new(n_features, exercise_encoder);
                self.enhanced_exercise_model = Some(model);
                println!("   Physics-based enhanced model ready!");
            } else {
                println!("    Enhanced exercise model retrained successfully!");
            }
        } else {
            println!("\n3. No XML file provided - creating physics-based enhanced model...");
            let exercise_encoder = self.create_exercise_type_encoder();
            let n_features = 19 + exercise_encoder.len();
            let model = EnhancedExercisePredictor::new(n_features, exercise_encoder);
            self.enhanced_exercise_model = Some(model);
            println!("   Physics-based enhanced model ready!");
        }
        
        println!("\n All models retrained successfully!");
        println!("Models are now ready with fresh training data.");
        
        Ok(())
    }
}

// Helper functions to convert between Array and Vec
fn array2_to_vec(arr: &Array2<f32>) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    for i in 0..arr.nrows() {
        let mut row = Vec::new();
        for j in 0..arr.ncols() {
            row.push(arr[[i, j]]);
        }
        result.push(row);
    }
    result
}

fn vec_to_array2(vec: &Vec<Vec<f32>>) -> Array2<f32> {
    if vec.is_empty() || vec[0].is_empty() {
        return Array2::zeros((0, 0));
    }
    let rows = vec.len();
    let cols = vec[0].len();
    let mut arr = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = vec[i][j];
        }
    }
    arr
}

fn array1_to_vec(arr: &Array1<f32>) -> Vec<f32> {
    arr.to_vec()
}

fn vec_to_array1(vec: &Vec<f32>) -> Array1<f32> {
    Array1::from_vec(vec.clone())
}

// Serializable versions of model structures
#[derive(Serialize, Deserialize)]
struct SerializableDailyModel {
    weights1: Vec<Vec<f32>>,
    biases1: Vec<f32>,
    weights2: Vec<Vec<f32>>,
    biases2: Vec<f32>,
    input_mins: Vec<f32>,
    input_maxs: Vec<f32>,
    target_min: f32,
    target_max: f32,
    input_size: usize,
    hidden_size: usize,
}

#[derive(Serialize, Deserialize)]
struct SerializableExerciseModel {
    weights1: Vec<Vec<f32>>,
    biases1: Vec<f32>,
    weights2: Vec<Vec<f32>>,
    biases2: Vec<f32>,
    weights3: Vec<Vec<f32>>,
    biases3: Vec<f32>,
    input_mins: Vec<f32>,
    input_maxs: Vec<f32>,
    target_min: f32,
    target_max: f32,
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
}

#[derive(Serialize, Deserialize)]
struct SerializableEnhancedModel {
    weights1: Vec<Vec<f32>>,
    biases1: Vec<f32>,
    weights2: Vec<Vec<f32>>,
    biases2: Vec<f32>,
    weights3: Vec<Vec<f32>>,
    biases3: Vec<f32>,
    weights4: Vec<Vec<f32>>,
    biases4: Vec<f32>,
    input_mins: Vec<f32>,
    input_maxs: Vec<f32>,
    target_min: f32,
    target_max: f32,
    exercise_type_encoder: HashMap<String, usize>,
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
    hidden_size3: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut calculator = CalorieCalculator::new();
    
    // Get XML path from command line or use default
    let xml_path = std::env::args().nth(1);
    
    if let Some(path) = &xml_path {
        println!("Attempting to load XML file: {}", path);
        
        // Check if file exists
        if !std::path::Path::new(path).exists() {
            println!("ERROR: XML file not found at: {}", path);
            println!("Please check the file path and try again.");
            println!();
            println!("Usage: {} <path_to_xml_file>", std::env::args().next().unwrap_or("calorie".to_string()));
            println!("Example: {} data/export.xml", std::env::args().next().unwrap_or("calorie".to_string()));
            return Ok(());
        }
    } else {
        println!("No XML file specified. Using physics-based model only.");
        println!();
        println!("To use XML training data:");
        println!("  {} <path_to_xml_file>", std::env::args().next().unwrap_or("calorie".to_string()));
        println!("  Example: {} data/export.xml", std::env::args().next().unwrap_or("calorie".to_string()));
        println!();
    }
    
    let xml_file = xml_path.as_deref();
    calculator.run(xml_file)
}

impl CalorieCalculator {
    // Add the missing methods here
    fn parse_xml_health_data(&self, xml_path: &str) -> Result<Vec<ProcessedExerciseData>, Box<dyn Error>> {
        let xml_content = std::fs::read_to_string(xml_path)?;
        println!("XML file size: {} bytes", xml_content.len());
        
        // Count elements for debugging
        let workout_count = xml_content.matches("<Workout").count();
        let record_count = xml_content.matches("<Record").count();
        println!("Found {} <Workout elements and {} <Record elements", workout_count, record_count);
        
        let mut processed_data = Vec::new();
        
        // Extract workout data using regex
        let workout_regex = regex::Regex::new(r#"<Workout[^>]*workoutActivityType="([^"]+)"[^>]*duration="([^"]+)"[^>]*startDate="([^"]+)"[^>]*>"#)?;        
        for workout_match in workout_regex.captures_iter(&xml_content) {
            println!("\n=== Processing Workout ===");
            let full_match = workout_match.get(0).unwrap();
            let workout_xml = &xml_content[full_match.start()..full_match.start()+200.min(xml_content.len()-full_match.start())];
            println!("Workout XML (first 200 chars): {}", workout_xml);
            
            let activity_type = workout_match.get(1).map_or("", |m| m.as_str());
            let duration_str = workout_match.get(2).map_or("", |m| m.as_str());
            let start_date = workout_match.get(3).map_or("", |m| m.as_str());
            
            println!("Raw attributes:");
            println!("    workoutActivityType: {}", activity_type);
            println!("    duration: {}", duration_str);
            println!("    totalEnergyBurned: 0");
            println!("    startDate: {}", start_date);
            
            if let Ok(duration_minutes) = duration_str.parse::<f32>() {
                println!("    Parsing duration: '{}'", duration_str);
                
                // Estimate calories based on workout type and duration
                let estimated_calories = self.estimate_calories_from_workout(activity_type, duration_minutes);
                println!("    Estimated calories based on workout type and duration: {}", estimated_calories);
                
                let exercise_type = self.map_workout_activity_type(activity_type);
                
                let processed_workout = ProcessedExerciseData {
                    exercise_type: exercise_type.clone(),
                    duration_minutes,
                    calories_burned: estimated_calories,
                    avg_heart_rate: Some(120.0 + (duration_minutes * 0.5).min(40.0)), // Estimate based on duration
                    estimated_age: 30.0,
                    estimated_weight: 70.0,
                    estimated_height: 170.0,
                    estimated_gender: true,
                    estimated_body_temp: 37.0,
                    estimated_resting_hr: 65.0,
                    estimated_max_hr: 190.0,
                    estimated_body_fat: 15.0,
                };
                
                println!("  Parsed values:");
                println!("    duration: {:.1} minutes", duration_minutes);
                println!("    calories: {}", estimated_calories);
                
                processed_data.push(processed_workout);
                println!("Successfully parsed workout: {} - {:.1}min - {}cal", exercise_type, duration_minutes, estimated_calories);
            }
        }
        
        println!("\nSuccessfully processed {} workouts from XML data", processed_data.len());
        Ok(processed_data)
    }
    
    fn convert_processed_to_enhanced(&self, processed_data: &[ProcessedExerciseData]) -> Vec<EnhancedExerciseData> {
        processed_data.iter().map(|data| {
            EnhancedExerciseData {
                gender: if data.estimated_gender { "Male".to_string() } else { "Female".to_string() },
                age: data.estimated_age,
                height: data.estimated_height,
                weight: data.estimated_weight,
                duration: data.duration_minutes,
                heart_rate: data.avg_heart_rate.unwrap_or(120.0),
                body_temp: data.estimated_body_temp,
                calories: data.calories_burned,
                exercise_type: data.exercise_type.clone(),
                resting_hr: data.estimated_resting_hr,
                max_hr: data.estimated_max_hr,
                body_fat_percent: data.estimated_body_fat,
                environmental_temp: 22.0,
                elevation: 0.0,
            }
        }).collect()
    }
    
    fn create_exercise_type_encoder(&self) -> HashMap<String, usize> {
        let mut encoder = HashMap::new();
        let exercise_types = vec![
            "Running", "Cycling", "Swimming", "Weight Training", "Walking", 
            "HIIT", "Rowing", "Elliptical", "Yoga", "Dance", "Other"
        ];
        
        for (i, exercise_type) in exercise_types.iter().enumerate() {
            encoder.insert(exercise_type.to_string(), i);
        }
        
        encoder
    }
    
    fn preprocess_enhanced_exercise_data(&self, data: &[EnhancedExerciseData], encoder: &HashMap<String, usize>) -> (Array2<f32>, Array1<f32>) {
        let n_samples = data.len();
        let n_base_features = 19; // Basic features
        let n_exercise_types = encoder.len();
        let n_features = n_base_features + n_exercise_types;
        
        let mut features = Array2::<f32>::zeros((n_samples, n_features));
        let mut targets = Array1::<f32>::zeros(n_samples);
        
        for (i, record) in data.iter().enumerate() {
            let mut feature_idx = 0;
            
            // Basic features
            features[[i, feature_idx]] = if record.gender.to_lowercase() == "male" { 1.0 } else { 0.0 };
            feature_idx += 1;
            features[[i, feature_idx]] = record.age;
            feature_idx += 1;
            features[[i, feature_idx]] = record.height;
            feature_idx += 1;
            features[[i, feature_idx]] = record.weight;
            feature_idx += 1;
            features[[i, feature_idx]] = record.duration;
            feature_idx += 1;
            features[[i, feature_idx]] = record.heart_rate;
            feature_idx += 1;
            features[[i, feature_idx]] = record.body_temp;
            feature_idx += 1;
            features[[i, feature_idx]] = record.resting_hr;
            feature_idx += 1;
            features[[i, feature_idx]] = record.max_hr;
            feature_idx += 1;
            features[[i, feature_idx]] = record.body_fat_percent;
            feature_idx += 1;
            features[[i, feature_idx]] = record.environmental_temp;
            feature_idx += 1;
            features[[i, feature_idx]] = record.elevation;

            
            // Exercise type encoding (one-hot)
            if let Some(&_type_idx) = encoder.get(&record.exercise_type) {
    
            }
            
            targets[i] = record.calories;
        }
        
        (features, targets)
    }
    
    fn estimate_calories_from_workout(&self, activity_type: &str, duration_minutes: f32) -> f32 {
        let base_calories_per_minute = match activity_type {
            s if s.contains("Running") => 15.0,
            s if s.contains("Cycling") => 10.0,
            s if s.contains("Swimming") => 12.0,
            s if s.contains("FunctionalStrengthTraining") => 6.25,
            s if s.contains("Walking") => 4.4,
            s if s.contains("HighIntensityIntervalTraining") => 15.0,
            s if s.contains("Rowing") => 11.0,
            s if s.contains("Elliptical") => 8.0,
            s if s.contains("Yoga") => 3.0,
            s if s.contains("Dance") => 7.0,
            _ => 5.0,
        };
        
        (base_calories_per_minute * duration_minutes).round()
    }
    
    fn map_workout_activity_type(&self, activity_type: &str) -> String {
        match activity_type {
            s if s.contains("Running") => "Running".to_string(),
            s if s.contains("Cycling") => "Cycling".to_string(),
            s if s.contains("Swimming") => "Swimming".to_string(),
            s if s.contains("FunctionalStrengthTraining") => "Weight Training".to_string(),
            s if s.contains("Walking") => "Walking".to_string(),
            s if s.contains("HighIntensityIntervalTraining") => "HIIT".to_string(),
            s if s.contains("Rowing") => "Rowing".to_string(),
            s if s.contains("Elliptical") => "Elliptical".to_string(),
            s if s.contains("Yoga") => "Yoga".to_string(),
            s if s.contains("Dance") => "Dance".to_string(),
            _ => {
                if !activity_type.is_empty() {
                    println!("      Unknown workout type: {}", activity_type);
                }
                "Other".to_string()
            }
        }
    }
}
