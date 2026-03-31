from training_and_modeling import predict_no_show_risk


print("\n--- No-Show Risk Prediction Tool ---")

while True:
    user_input = input("Enter patient ID (or 'q' to quit): ")
    
    if user_input.lower() == 'q':
        print("Exiting program.")
        break
    
    try:
        patient_id = int(user_input)
        result = predict_no_show_risk(patient_id)
        
        if isinstance(result, dict):
            print(f"\nPatient ID: {result['patient_id']}")
            print(f"No-show probability: {result['no_show_probability']:.1%}")
            print(f"Risk level: {result['risk_level']}")
            
            # ✅ Print risk factors if present
            if "key_risk_factors" in result:
                print("Key risk factors:")
                for factor in result["key_risk_factors"]:
                    print(f"  - {factor}")
            
            # ✅ Print recommendations if present
            if "recommended_actions" in result:
                print("Recommended actions:")
                for action in result["recommended_actions"]:
                    print(f"  - {action}")
                
        else:
            print(result)
            
    except:
        print("Please enter a valid patient ID.")
