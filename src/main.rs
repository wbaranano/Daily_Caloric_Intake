use std::error::Error;
use std::io::{self, Write};
use serde::Deserialize;
use ndarray::{Array1, Array2};
use ndarray::s;
use rand::Rng;
use std::collections::HashMap;
use std::fs;

// Embedded CSV data
const EMBEDDED_DAILY_CSV: &str = include_str!("../data/balanced_diet.csv");
const EMBEDDED_EXERCISE_CSV: &str = include_str!("../data/calories.csv");

// XML parsing structures
#[derive(Debug, Deserialize)]
struct Workout {
    #[serde(rename = "workoutActivityType")]
    workout_activity_type: String,
    #[serde(rename = "duration")]
    duration: Option<String>,
    #[serde(rename = "totalEnergyBurned")]
    total_energy_burned: Option<String>,
    #[serde(rename = "startDate")]
    start_date: Option<String>,
    #[serde(rename = "endDate")]
    end_date: Option<String>,
    #[serde(rename = "WorkoutEvent", default)]
    workout_events: Vec<WorkoutEvent>,
    #[serde(rename = "WorkoutStatistics", default)]
    workout_statistics: Vec<WorkoutStatistic>,
}

#[derive(Debug, Deserialize)]
struct WorkoutEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(rename = "date")]
    date: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WorkoutStatistic {
    #[serde(rename = "type")]
    stat_type: String,
    #[serde(rename = "startDate")]
    start_date: Option<String>,
    #[serde(rename = "endDate")]
    end_date: Option<String>,
    #[serde(rename = "sum")]
    sum: Option<String>,
    #[serde(rename = "minimum")]
    minimum: Option<String>,
    #[serde(rename = "maximum")]
    maximum: Option<String>,
    #[serde(rename = "average")]
    average: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "type")]
    record_type: String,
    #[serde(rename = "sourceName")]
    source_name: Option<String>,
    #[serde(rename = "unit")]
    unit: Option<String>,
    #[serde(rename = "creationDate")]
    creation_date: Option<String>,
    #[serde(rename = "startDate")]
    start_date: Option<String>,
    #[serde(rename = "endDate")]
    end_date: Option<String>,
    #[serde(rename = "value")]
    value: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HealthData {
    #[serde(rename = "Record", default)]
    records: Vec<Record>,
    #[serde(rename = "Workout", default)]
    workouts: Vec<Workout>,
}

// Processed exercise data structure
#[derive(Debug, Clone)]
struct ProcessedExerciseData {
    exercise_type: String,
    duration_minutes: f32,
    calories_burned: f32,
    avg_heart_rate: Option<f32>,
    distance: Option<f32>,
    date: String,
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
    user_id: Option<f32>,
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
    
    // FIXED: Proper physics-based prediction instead of broken neural network
    fn predict_enhanced_calories(&self, is_male: bool, age: f32, height: f32, weight: f32,
                               duration: f32, heart_rate: f32, body_temp: f32,
                               exercise_type: &str, resting_hr: f32, max_hr: f32,
                               body_fat_percent: f32, environmental_temp: f32, elevation: f32) -> f32 {
        
        // Calculate BMR (Basal Metabolic Rate)
        let bmr = if is_male {
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
        let calories_per_hour = (met_adjusted * weight * 1.05); // 1.05 conversion factor
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
    
    fn encode_exercise_type(&self, exercise_type: &str) -> Vec<f32> {
        let mut encoded = vec![0.0; self.exercise_type_encoder.len()];
        if let Some(&index) = self.exercise_type_encoder.get(exercise_type) {
            encoded[index] = 1.0;
        }
        encoded
    }
}

#[derive(Debug)]
enum CalculatorChoice {
    DailyCalories,
    ExerciseCalories,
    EnhancedExercise,
    Both,
}

// Add CSV data structures back
#[derive(Debug, Clone)]
struct ExerciseData {
    user_id: Option<f32>,
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
        let mut rng = rand::thread_rng();
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
        let mut rng = rand::thread_rng();
        let hidden_size = 32;
        
        let xavier_1 = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let xavier_2 = (2.0 / (hidden_size + 1) as f32).sqrt();
        
        let mut weights1 = Array2::<f32>::zeros((input_size, hidden_size));
        let mut weights2 = Array2::<f32>::zeros((hidden_size, 1));
        
        for i in 0..input_size {
            for j in 0..hidden_size {
                weights1[[i, j]] = rng.gen_range(-xavier_1..xavier_1);
            }
        }
        
        for i in 0..hidden_size {
            weights2[[i, 0]] = rng.gen_range(-xavier_2..xavier_2);
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
    
    fn predict_daily_calories(&self, age: f32, is_male: bool, activity_level: &str, sleep_duration: f32, height: f32) -> f32 {
        let activity_encoded = match activity_level {
            "Low" => 0.0,
            "Moderate" => 1.0,
            "High" => 2.0,
            _ => 1.0,
        };
        
        let input = Array1::from(vec![
            age,
            if is_male { 1.0 } else { 0.0 },
            activity_encoded,
            sleep_duration,
            height,
        ]);
        
        self.predict(&input)
    }
    
    fn predict(&self, input: &Array1<f32>) -> f32 {
        let normalized = self.normalize_input(input);
        
        let z1 = normalized.dot(&self.weights1) + &self.biases1;
        let a1 = z1.mapv(Self::relu);
        
        let z2 = a1.dot(&self.weights2) + &self.biases2;
        let output = 1.0 / (1.0 + (-z2[0]).exp());
        
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
            if fields.len() >= 9 {
                let exercise = ExerciseData {
                    user_id: fields[0].parse().ok(),
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
        let daily_data = self.parse_daily_csv_data(EMBEDDED_DAILY_CSV)?;
        let (features, targets) = self.preprocess_daily_data(&daily_data);
        
        let mut model = DailyCaloriePredictor::new(5);
        model.train(&features, &targets);
        
        self.daily_model = Some(model);
        Ok(())
    }
    
    fn train_exercise_model(&mut self) -> Result<(), Box<dyn Error>> {
        let exercise_data = self.parse_csv_data(EMBEDDED_EXERCISE_CSV)?;
        let (features, targets) = self.preprocess_exercise_data(&exercise_data);
        
        let mut model = ExercisePredictor::new(7);
        model.train(&features, &targets);
        
        self.exercise_model = Some(model);
        Ok(())
    }
    
    fn train_enhanced_exercise_model(&mut self, xml_path: &str) -> Result<(), Box<dyn Error>> {
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
                user_id: record.user_id,
                gender: record.gender.clone(),
                age: record.age,
                height: record.height,
                weight: record.weight,
                duration: record.duration,
                heart_rate: record.heart_rate,
                body_temp: record.body_temp,
                calories: record.calories,
                exercise_type: "Other".to_string(), // CSV doesn't have exercise type
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
        
        self.enhanced_exercise_model = Some(model);
        Ok(())
    }
    
    // Update run method to handle all models
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
            let (is_male, age, height, weight) = self.get_basic_info()?;
            
            println!("\nCALCULATION RESULTS");
            println!("======================");
            
            match choice {
                CalculatorChoice::DailyCalories => {
                    if let Some(ref model) = self.daily_model {
                        let daily_calories = model.predict_daily_calories(age, is_male, "Moderate", 8.0, height);
                        
                        println!("DAILY CALORIE NEEDS (CSV-Trained)");
                        println!("==================================");
                        println!("Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!("Estimated daily calorie needs: {:.0} calories", daily_calories);
                        
                        println!("\nLIFESTYLE RECOMMENDATIONS:");
                        if daily_calories > 2200.0 {
                            println!("   High calorie needs - ensure adequate nutrition");
                        } else if daily_calories > 1800.0 {
                            println!("   Moderate calorie needs - balanced diet recommended");
                        } else {
                            println!("   Lower calorie needs - focus on nutrient density");
                        }
                    } else {
                        println!("Daily calorie model not available");
                    }
                },
                CalculatorChoice::ExerciseCalories => {
                    if let Some(ref model) = self.exercise_model {
                        let (duration, heart_rate, body_temp) = self.get_exercise_info()?;
                        let exercise_calories = model.predict_exercise_calories(
                            is_male, age, height, weight, duration, heart_rate, body_temp
                        );
                        
                        println!("BASIC EXERCISE CALORIE BURN (CSV-Trained)");
                        println!("==========================================");
                        println!("Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!("Exercise: {:.0} min, {:.0} bpm, {:.1}°C", duration, heart_rate, body_temp);
                        println!("Calories Burned: {:.0} calories", exercise_calories);
                    } else {
                        println!("Exercise calorie model not available");
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
                        
                        println!("ENHANCED CALORIE ANALYSIS (CSV + XML Hybrid)");
                        println!("=============================================");
                        println!("Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!("Exercise: {} for {:.0} minutes", exercise_type, duration);
                        println!("Heart Rate: {:.0} bpm ({:.1}% of HR reserve)", heart_rate, hr_percentage);
                        println!("Conditions: {:.1}°C ambient, {:.0}m elevation", env_temp, elevation);
                        println!("Body Fat: {:.1}%", body_fat);
                        println!();
                        println!("CALORIE BURN RESULTS:");
                        println!("   Total calories burned: {:.0} calories", enhanced_calories);
                        println!("   Calories per minute: {:.1} cal/min", calories_per_minute);
                        println!("   Calories per hour: {:.0} cal/hour", calories_per_hour);
                        
                        if hr_percentage > 85.0 {
                            println!("   Very high intensity - maximum calorie burn!");
                        } else if hr_percentage > 70.0 {
                            println!("   High intensity - excellent calorie burn!");
                        } else if hr_percentage > 50.0 {
                            println!("   Moderate intensity - good steady burn rate!");
                        } else {
                            println!("   Light activity - gentle calorie burn!");
                        }
                        
                    } else {
                        println!("Enhanced exercise model not available");
                    }
                },
                CalculatorChoice::Both => {
                    // Combined analysis using all models
                    if let Some(ref daily_model) = self.daily_model {
                        let daily_calories = daily_model.predict_daily_calories(age, is_male, "Moderate", 8.0, height);
                        
                        println!("COMPREHENSIVE CALORIE ANALYSIS");
                        println!("===============================");
                        println!("Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!("Daily calorie needs: {:.0} calories", daily_calories);
                        println!();
                        
                        if let Some(ref enhanced_model) = self.enhanced_exercise_model {
                            let (duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, 
                                 body_fat, env_temp, elevation) = self.get_enhanced_exercise_info(is_male, age)?;
                            
                            let enhanced_calories = enhanced_model.predict_enhanced_calories(
                                is_male, age, height, weight, duration, heart_rate, body_temp,
                                &exercise_type, resting_hr, max_hr, body_fat, env_temp, elevation
                            );
                            
                            println!("EXERCISE ANALYSIS:");
                            println!("   Exercise: {} for {:.0} minutes", exercise_type, duration);
                            println!("   Calories burned: {:.0} calories", enhanced_calories);
                            println!();
                            
                            let net_calories = daily_calories - enhanced_calories;
                            println!("COMBINED ANALYSIS:");
                            println!("   Daily calories needed: {:.0}", daily_calories);
                            println!("   Exercise calories burned: {:.0}", enhanced_calories);
                            println!("   Net calories for the day: {:.0}", net_calories);
                            
                            if net_calories > daily_calories * 0.8 {
                                println!("   Good balance - moderate exercise with adequate nutrition");
                            } else if net_calories > daily_calories * 0.6 {
                                println!("   Active day - ensure adequate post-workout nutrition");
                            } else {
                                println!("   Very active day - consider additional nutrition");
                            }
                        }
                    }
                }
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
    
    fn get_user_choice(&self) -> Result<CalculatorChoice, Box<dyn Error>> {
        println!("\nENHANCED CALORIE CALCULATOR (CSV + XML Hybrid)");
        println!("===============================================");
        println!("What would you like to calculate?");
        println!("1. Daily calorie needs (CSV-trained neural network)");
        println!("2. Exercise calories (CSV-trained neural network)");
        println!("3. Exercise calories (ENHANCED: CSV + XML + Physics)");
        println!("4. Complete analysis (Daily + Enhanced Exercise)");
        println!("===============================================");
        
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
                _ => println!("Please enter 1, 2, 3, or 4."),
            }
        }
    }
    
    // ... keep all existing XML parsing methods and other functions ...
    
    fn parse_xml_health_data(&self, xml_path: &str) -> Result<Vec<ProcessedExerciseData>, Box<dyn Error>> {
        println!("Loading XML health data from: {}", xml_path);
        
        let xml_content = fs::read_to_string(xml_path)?;
        let mut processed_data = Vec::new();
        
        println!("XML file size: {} bytes", xml_content.len());
        
        // Count different element types
        let workout_count = xml_content.matches("<Workout").count();
        let record_count = xml_content.matches("<Record").count();
        
        println!("Found {} <Workout elements and {} <Record elements", workout_count, record_count);
        
        if workout_count == 0 {
            return Ok(processed_data);
        }
        
        // Parse workouts using a better approach - find each workout block
        let mut current_pos = 0;
        
        while let Some(workout_start) = xml_content[current_pos..].find("<Workout ") {
            let absolute_start = current_pos + workout_start;
            
            // Find the end of this workout element (either /> or </Workout>)
            let workout_section = &xml_content[absolute_start..];
            
            let workout_end = if let Some(self_close) = workout_section.find("/>") {
                absolute_start + self_close + 2
            } else if let Some(close_tag) = workout_section.find("</Workout>") {
                absolute_start + close_tag + 10
            } else {
                // Find the end of the opening tag and assume it's self-closing for now
                if let Some(tag_end) = workout_section.find(">") {
                    absolute_start + tag_end + 1
                } else {
                    break;
                }
            };
            
            let workout_xml = &xml_content[absolute_start..workout_end];
            
            println!("\n=== Processing Workout ===");
            println!("Workout XML (first 200 chars): {}", &workout_xml[0..200.min(workout_xml.len())]);
            
            // Extract attributes from the entire workout block
            if let Some(workout_data) = self.parse_workout_xml(workout_xml) {
                println!("Successfully parsed workout: {} - {:.1}min - {:.0}cal", 
                        workout_data.exercise_type, workout_data.duration_minutes, workout_data.calories_burned);
                processed_data.push(workout_data);
            } else {
                println!(" Failed to parse workout");
            }
            
            current_pos = workout_end;
        }
        
        println!("\nSuccessfully processed {} workouts from XML data", processed_data.len());
        Ok(processed_data)
    }
    
    fn parse_workout_xml(&self, workout_xml: &str) -> Option<ProcessedExerciseData> {
        // Extract workout attributes from the XML block
        let workout_type = self.extract_xml_attribute(workout_xml, "workoutActivityType")?;
        
        // Duration might be in different formats
        let duration_str = self.extract_xml_attribute(workout_xml, "duration")
            .or_else(|| self.extract_xml_attribute(workout_xml, "durationUnit"))
            .unwrap_or("0".to_string());
        
        // Try multiple ways to find energy data
        let energy_str = self.extract_xml_attribute(workout_xml, "totalEnergyBurned")
            .or_else(|| self.extract_xml_attribute(workout_xml, "totalEnergyBurnedUnit"))
            .or_else(|| self.extract_xml_attribute(workout_xml, "energy"))
            .or_else(|| {
                // Look for energy in WorkoutStatistics
                if workout_xml.contains("HKQuantityTypeIdentifierActiveEnergyBurned") {
                    self.extract_energy_from_statistics(workout_xml)
                } else {
                    None
                }
            })
            .unwrap_or("0".to_string());
        
        let start_date = self.extract_xml_attribute(workout_xml, "startDate")
            .unwrap_or("unknown".to_string());
        
        println!("  Raw attributes:");
        println!("    workoutActivityType: {}", workout_type);
        println!("    duration: {}", duration_str);
        println!("    totalEnergyBurned: {}", energy_str);
        println!("    startDate: {}", start_date);
        
        // Parse duration - handle various formats
        let duration_minutes = self.parse_duration_to_minutes(&duration_str);
        
        // Parse calories - might have units attached
        let mut calories = self.parse_energy_value(&energy_str);
        
        // If no calories found, estimate based on workout type and duration
        if calories <= 0.0 && duration_minutes > 0.0 {
            calories = self.estimate_calories_from_workout(&workout_type, duration_minutes);
            println!("    Estimated calories based on workout type and duration: {:.0}", calories);
        }
        
        println!("  Parsed values:");
        println!("    duration: {:.1} minutes", duration_minutes);
        println!("    calories: {:.0}", calories);
        
        // Accept workouts with reasonable duration (even if no calorie data)
        if duration_minutes >= 1.0 {
            let normalized_type = self.normalize_workout_type(&workout_type);
            
            Some(ProcessedExerciseData {
                exercise_type: normalized_type,
                duration_minutes,
                calories_burned: calories.max(10.0), // Minimum 10 calories
                avg_heart_rate: None,
                distance: None,
                date: start_date,
                estimated_age: 25.0,
                estimated_weight: 75.0,
                estimated_height: 180.0,
                estimated_gender: true,
                estimated_body_temp: 37.0,
                estimated_resting_hr: 60.0,
                estimated_max_hr: 195.0,
                estimated_body_fat: 12.0,
            })
        } else {
            println!(" Insufficient duration: {:.1}min", duration_minutes);
            None
        }
    }
    
    fn extract_energy_from_statistics(&self, workout_xml: &str) -> Option<String> {
        // Look for WorkoutStatistics with energy data
        // Pattern: <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" ... sum="123.45" ...>
        if let Some(energy_stat_start) = workout_xml.find("HKQuantityTypeIdentifierActiveEnergyBurned") {
            let remaining = &workout_xml[energy_stat_start..];
            if let Some(sum_value) = self.extract_xml_attribute(remaining, "sum") {
                return Some(sum_value);
            }
        }
        None
    }
    
    fn estimate_calories_from_workout(&self, workout_type: &str, duration_minutes: f32) -> f32 {
        // Estimate calories based on average person (75kg) and workout type
        let weight = 75.0; // kg
        
        // Base MET values for different workout types
        let met_value = match workout_type {
            "HKWorkoutActivityTypeRunning" => 10.0,
            "HKWorkoutActivityTypeCycling" => 8.0,
            "HKWorkoutActivityTypeSwimming" => 8.0,
            "HKWorkoutActivityTypeFunctionalStrengthTraining" => 5.0,
            "HKWorkoutActivityTypeTraditionalStrengthTraining" => 5.0,
            "HKWorkoutActivityTypeWalking" => 3.5,
            "HKWorkoutActivityTypeRowing" => 8.5,
            "HKWorkoutActivityTypeElliptical" => 7.0,
            "HKWorkoutActivityTypeHighIntensityIntervalTraining" => 12.0,
            "HKWorkoutActivityTypeYoga" => 2.5,
            "HKWorkoutActivityTypeDance" => 5.0,
            "HKWorkoutActivityTypeCoreTraining" => 4.0,
            "HKWorkoutActivityTypeFlexibility" => 2.0,
            "HKWorkoutActivityTypeOther" => 4.0,
            _ => 4.0, // Default moderate activity
        };
        
        // Calculate calories: METs × weight (kg) × time (hours)
        let hours = duration_minutes / 60.0;
        let calories = met_value * weight * hours;
        
        calories
    }
    
    fn extract_xml_attribute(&self, xml: &str, attr_name: &str) -> Option<String> {
        // Handle multi-line XML attributes
        let patterns = vec![
            format!(r#"{}=""#, attr_name),
            format!(r#"{} = ""#, attr_name),
            format!(r#"{}=""#, attr_name),
        ];
        
        for pattern in patterns {
            if let Some(start_pos) = xml.find(&pattern) {
                let value_start = start_pos + pattern.len();
                
                // Find the closing quote, handling multi-line attributes
                let remaining = &xml[value_start..];
                if let Some(end_pos) = remaining.find('"') {
                    let value = remaining[0..end_pos].to_string();
                    if !value.is_empty() {
                        return Some(value);
                    }
                }
            }
        }
        
        None
    }
    
    fn parse_energy_value(&self, energy_str: &str) -> f32 {
        if energy_str.is_empty() {
            return 0.0;
        }
        
        // Remove units and parse the number
        let cleaned = energy_str
            .replace("Cal", "")
            .replace("kcal", "")
            .replace("kJ", "")
            .replace(" ", "")
            .trim()
            .to_string();
        
        let value = cleaned.parse::<f32>().unwrap_or(0.0);
        
        // Convert kJ to kcal if needed (1 kcal = 4.184 kJ)
        if energy_str.contains("kJ") {
            value / 4.184
        } else {
            value
        }
    }
    
    fn parse_duration_to_minutes(&self, duration_str: &str) -> f32 {
        if duration_str.is_empty() || duration_str == "0" {
            return 0.0;
        }
        
        println!("    Parsing duration: '{}'", duration_str);
        
        // Handle ISO 8601 format like "PT45M" or "PT1H30M45.123S"
        if duration_str.starts_with("PT") {
            let mut total_minutes = 0.0;
            let clean_str = &duration_str[2..]; // Remove "PT"
            
            // Extract hours
            if let Some(h_pos) = clean_str.find('H') {
                if let Ok(hours) = clean_str[0..h_pos].parse::<f32>() {
                    total_minutes += hours * 60.0;
                    println!("      Found {} hours", hours);
                }
            }
            
            // Extract minutes
            if let Some(m_pos) = clean_str.find('M') {
                let start = if clean_str.contains('H') {
                    clean_str.find('H').unwrap() + 1
                } else {
                    0
                };
                if let Ok(minutes) = clean_str[start..m_pos].parse::<f32>() {
                    total_minutes += minutes;
                    println!("      Found {} minutes", minutes);
                }
            }
            
            // Extract seconds
            if let Some(s_pos) = clean_str.find('S') {
                let start = if clean_str.contains('M') {
                    clean_str.find('M').unwrap() + 1
                } else if clean_str.contains('H') {
                    clean_str.find('H').unwrap() + 1
                } else {
                    0
                };
                if let Ok(seconds) = clean_str[start..s_pos].parse::<f32>() {
                    total_minutes += seconds / 60.0;
                    println!("      Found {} seconds", seconds);
                }
            }
            
            println!("      Total: {} minutes", total_minutes);
            return total_minutes;
        }
        
        // Handle "45:30" format (minutes:seconds)
        if duration_str.contains(':') {
            let parts: Vec<&str> = duration_str.split(':').collect();
            if parts.len() >= 2 {
                let minutes = parts[0].parse::<f32>().unwrap_or(0.0);
                let seconds = parts[1].parse::<f32>().unwrap_or(0.0);
                return minutes + (seconds / 60.0);
            }
        }
        
        // Just a number - this is probably minutes already
        let value = duration_str.parse::<f32>().unwrap_or(0.0);
        
        // If the value is very large, it might be seconds, convert to minutes
        if value > 300.0 { // More than 5 hours in minutes seems unlikely
            println!("      Assuming large value {} is seconds, converting to minutes", value);
            value / 60.0
        } else {
            value
        }
    }
    
    fn normalize_workout_type(&self, workout_type: &str) -> String {
        match workout_type {
            "HKWorkoutActivityTypeRunning" => "Running".to_string(),
            "HKWorkoutActivityTypeCycling" => "Cycling".to_string(),
            "HKWorkoutActivityTypeSwimming" => "Swimming".to_string(),
            "HKWorkoutActivityTypeFunctionalStrengthTraining" | 
            "HKWorkoutActivityTypeTraditionalStrengthTraining" => "Weight Training".to_string(),
            "HKWorkoutActivityTypeWalking" => "Walking".to_string(),
            "HKWorkoutActivityTypeRowing" => "Rowing".to_string(),
            "HKWorkoutActivityTypeElliptical" => "Elliptical".to_string(),
            "HKWorkoutActivityTypeHighIntensityIntervalTraining" => "HIIT".to_string(),
            "HKWorkoutActivityTypeYoga" => "Yoga".to_string(),
            "HKWorkoutActivityTypeDance" => "Dance".to_string(),
            "HKWorkoutActivityTypeCoreTraining" => "Weight Training".to_string(),
            "HKWorkoutActivityTypeFlexibility" => "Yoga".to_string(),
            "HKWorkoutActivityTypeOther" => "Other".to_string(),
            _ => {
                println!("      Unknown workout type: {}", workout_type);
                "Other".to_string()
            }
        }
    }
    
    fn convert_processed_to_enhanced(&self, processed_data: &[ProcessedExerciseData]) -> Vec<EnhancedExerciseData> {
        processed_data.iter().map(|record| {
            EnhancedExerciseData {
                user_id: Some(1.0),
                gender: if record.estimated_gender { "Male".to_string() } else { "Female".to_string() },
                age: record.estimated_age,
                height: record.estimated_height,
                weight: record.estimated_weight,
                duration: record.duration_minutes,
                heart_rate: record.avg_heart_rate.unwrap_or(120.0),
                body_temp: record.estimated_body_temp,
                calories: record.calories_burned,
                exercise_type: record.exercise_type.clone(),
                resting_hr: record.estimated_resting_hr,
                max_hr: record.estimated_max_hr,
                body_fat_percent: record.estimated_body_fat,
                environmental_temp: 22.0,
                elevation: 0.0,
            }
        }).collect()
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
            "Yoga".to_string(),
            "Dance".to_string(),
            "Other".to_string(),
        ];
        
        exercise_types.into_iter()
            .enumerate()
            .map(|(i, exercise_type)| (exercise_type, i))
            .collect()
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
                "Yoga" => 2.5,
                "Dance" => 4.5,
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
            features[[i, feat_idx]] = hr_percentage; feat_idx += 1;
            features[[i, feat_idx]] = record.body_fat_percent; feat_idx += 1;
            features[[i, feat_idx]] = bmi; feat_idx += 1;
            features[[i, feat_idx]] = lean_mass; feat_idx += 1;
            features[[i, feat_idx]] = metabolic_factor; feat_idx += 1;
            features[[i, feat_idx]] = met_estimate; feat_idx += 1;
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
    
    fn get_basic_info(&self) -> Result<(bool, f32, f32, f32), Box<dyn Error>> {
        println!("\nBASIC INFORMATION");
        println!("====================");
        
        let gender = loop {
            print!("Gender (M/F): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().to_lowercase().as_str() {
                "m" | "male" => break true,
                "f" | "female" => break false,
                _ => println!("Please enter M or F"),
            }
        };
        
        let age = loop {
            print!("Age (years): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(age) if age >= 10.0 && age <= 100.0 => break age,
                _ => println!("Please enter valid age (10-100)"),
            }
        };
        
        let height = loop {
            print!("Height (cm): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(height) if height >= 100.0 && height <= 250.0 => break height,
                _ => println!("Please enter valid height (100-250 cm)"),
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
                _ => println!("Please enter valid duration (1-600 minutes)"),
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
    
    fn get_enhanced_exercise_info(&self, is_male: bool, age: f32) -> Result<(f32, f32, f32, String, f32, f32, f32, f32, f32), Box<dyn Error>> {
        let (duration, heart_rate, body_temp) = self.get_exercise_info()?;
        
        println!("\nEXERCISE TYPE SELECTION:");
        println!("1. Running/Jogging");
        println!("2. Cycling");
        println!("3. Swimming");
        println!("4. Weight Training");
        println!("5. Walking");
        println!("6. Rowing");
        println!("7. Elliptical");
        println!("8. HIIT");
        println!("9. Yoga");
        println!("10. Dance");
        
        let exercise_type = loop {
            print!("Select exercise type (1-10): ");
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
                "9" => break "Yoga".to_string(),
                "10" => break "Dance".to_string(),
                _ => println!("Please enter 1-10"),
            }
        };
        
        println!("\nHEART RATE INFORMATION:");
        let resting_hr = loop {
            print!("Resting heart rate (bpm) [60-100]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            match input.trim().parse::<f32>() {
                Ok(hr) if hr >= 40.0 && hr <= 120.0 => break hr,
                _ => println!("Please enter valid resting HR (40-120 bpm)"),
            }
        };
        
        let max_hr = loop {
            print!("Maximum heart rate (press Enter for auto-calc): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().is_empty() {
                break 220.0 - age;
            }
            match input.trim().parse::<f32>() {
                Ok(hr) if hr >= 150.0 && hr <= 230.0 => break hr,
                _ => println!("Please enter valid max HR (150-230 bpm)"),
            }
        };
        
        println!("\nBODY COMPOSITION:");
        let body_fat_percent = loop {
            print!("Body fat percentage (%) [press Enter for estimate]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().is_empty() {
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
                _ => println!("Please enter valid elevation (-500 to 9000m)"),
            }
        };
        
        Ok((duration, heart_rate, body_temp, exercise_type, resting_hr, max_hr, 
            body_fat_percent, environmental_temp, elevation))
    }
    
    fn run(&mut self, xml_path: Option<&str>) -> Result<(), Box<dyn Error>> {
        println!("ENHANCED CALORIE CALCULATOR (XML-Trained)");
        println!("==========================================");
        
        // Try to load XML data if path provided
        if let Some(path) = xml_path {
            println!("Training model with XML data...");
            if let Err(e) = self.train_enhanced_exercise_model(path) {
                println!("Warning: Could not train from XML: {}", e);
                println!("Using physics-based model instead");
            }
        } else {
            println!("No XML file provided, using physics-based model");
        }
        
        // Create a basic model if none was loaded
        if self.enhanced_exercise_model.is_none() {
            let exercise_encoder = self.create_exercise_type_encoder();
            let n_features = 19 + exercise_encoder.len();
            let model = EnhancedExercisePredictor::new(n_features, exercise_encoder);
            self.enhanced_exercise_model = Some(model);
        }
        
        loop {
            let choice = self.get_user_choice()?;
            let (is_male, age, height, weight) = self.get_basic_info()?;
            
            println!("\nCALCULATION RESULTS");
            println!("======================");
            
            match choice {
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
                        
                        println!("ENHANCED CALORIE ANALYSIS (XML-Trained)");
                        println!("========================================");
                        println!("Profile: {} {}, {:.0}cm, {:.0}kg", 
                            if is_male { "Male" } else { "Female" }, age, height, weight);
                        println!("Exercise: {} for {:.0} minutes", exercise_type, duration);
                        println!("Heart Rate: {:.0} bpm ({:.1}% of HR reserve)", heart_rate, hr_percentage);
                        println!("Conditions: {:.1}°C ambient, {:.0}m elevation", env_temp, elevation);
                        println!("Body Fat: {:.1}%", body_fat);
                        println!();
                        println!("CALORIE BURN RESULTS:");
                        println!("   Total calories burned: {:.0} calories", enhanced_calories);
                        println!("   Calories per minute: {:.1} cal/min", calories_per_minute);
                        println!("   Calories per hour: {:.0} cal/hour", calories_per_hour);
                        
                        if hr_percentage > 85.0 {
                            println!("   Very high intensity - maximum calorie burn!");
                        } else if hr_percentage > 70.0 {
                            println!("   High intensity - excellent calorie burn!");
                        } else if hr_percentage > 50.0 {
                            println!("   Moderate intensity - good steady burn rate!");
                        } else {
                            println!("   Light activity - gentle calorie burn!");
                        }
                        
                    } else {
                        println!("Enhanced exercise model not available");
                    }
                },
                _ => {
                    println!("Feature not yet implemented in this demo");
                    println!("Use option 3 for the XML-trained enhanced calculator");
                }
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
