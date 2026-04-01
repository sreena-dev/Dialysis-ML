def calculate_uf_rate(pre_weight_kg: float, dry_weight_kg: float, time_hr: float) -> float:
    """
    Calculates the Ultrafiltration (UF) rate in dialysis.
    
    Formula:
    UF Rate (mL/hr/kg) = (Weight Gain (kg) * 1000) / (Time (hr) * Target Dry Weight (kg))
    Weight Gain = Pre-dialysis weight - Target dry weight
    
    Args:
        pre_weight_kg: Pre-dialysis weight in kilograms.
        dry_weight_kg: Target dry weight in kilograms.
        time_hr: Treatment time in hours.
        
    Returns:
        float: Calculated UF Rate in mL/hr/kg.
    """
    if time_hr <= 0 or dry_weight_kg <= 0:
        return 0.0
        
    weight_gain_kg = pre_weight_kg - dry_weight_kg
    fluid_to_remove_ml = weight_gain_kg * 1000.0
    uf_rate_ml_hr = fluid_to_remove_ml / time_hr
    uf_rate_ml_hr_kg = uf_rate_ml_hr / dry_weight_kg
    
    return uf_rate_ml_hr_kg
