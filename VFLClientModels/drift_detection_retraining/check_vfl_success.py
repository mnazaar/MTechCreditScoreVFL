#!/usr/bin/env python3
"""
VFL Training Success Verification Script

This script helps manually verify if VFL training actually succeeded,
even when the automated pipeline reports failure.

Usage:
    python check_vfl_success.py
"""

import os
import time
from datetime import datetime

def check_vfl_model_files():
    """Check if VFL model files exist and when they were last modified"""
    
    print("🔍 VFL Training Success Verification")
    print("=" * 50)
    
    # Critical VFL model files that indicate successful training
    vfl_model_files = {
        'Main VFL Model': 'VFLClientModels/saved_models/vfl_automl_xgboost_simple_model.keras',
        'Homomorphic Model': 'VFLClientModels/saved_models/vfl_automl_xgboost_homomorp_model.keras',
        'Best Hyperparameters': 'VFLClientModels/saved_models/best_hyperparameters_homoenc_dp.pkl',
        'Prediction Cache': 'VFLClientModels/saved_models/prediction_cache_homoenc_dp.pkl',
        'Auto Loans Scaler': 'VFLClientModels/saved_models/auto_loans_scaler_homoenc_dp.pkl',
        'Digital Bank Scaler': 'VFLClientModels/saved_models/digital_bank_scaler_homoenc_dp.pkl',
        'Home Loans Scaler': 'VFLClientModels/saved_models/home_loans_scaler_homoenc_dp.pkl'
    }
    
    print(f"📁 Checking VFL model files:")
    print()
    
    existing_files = []
    recent_files = []
    current_time = time.time()
    
    for description, file_path in vfl_model_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
            modified_time = datetime.fromtimestamp(file_mtime)
            minutes_ago = (current_time - file_mtime) / 60
            hours_ago = minutes_ago / 60
            
            print(f"✅ {description}")
            print(f"   📄 File: {file_path}")
            print(f"   📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print(f"   🕒 Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if hours_ago < 1:
                print(f"   ⏰ Last modified: {minutes_ago:.0f} minutes ago (VERY RECENT)")
                recent_files.append((description, minutes_ago))
            elif hours_ago < 24:
                print(f"   ⏰ Last modified: {hours_ago:.1f} hours ago")
                if hours_ago < 12:  # Consider files modified in last 12 hours as "recent"
                    recent_files.append((description, minutes_ago))
            else:
                days_ago = hours_ago / 24
                print(f"   ⏰ Last modified: {days_ago:.1f} days ago")
            
            print()
            existing_files.append(description)
        else:
            print(f"❌ {description}")
            print(f"   📄 File: {file_path}")
            print(f"   ❗ Status: FILE NOT FOUND")
            print()
    
    # Summary analysis
    print("📊 Analysis Summary:")
    print("=" * 30)
    print(f"✅ Files found: {len(existing_files)}/{len(vfl_model_files)}")
    print(f"🕒 Recently modified files: {len(recent_files)}")
    
    if len(existing_files) == len(vfl_model_files):
        print("🎉 All VFL model files are present!")
        
        if recent_files:
            print("⚡ Recent modifications detected:")
            for desc, minutes in recent_files:
                print(f"   - {desc} ({minutes:.0f} minutes ago)")
            print()
            print("✅ VERDICT: VFL training likely SUCCEEDED recently")
            print("   The presence of all model files with recent timestamps")
            print("   strongly suggests successful training completion.")
        else:
            print("📅 No recent modifications detected")
            print("⚠️  VERDICT: VFL models exist but may be from previous training")
            print("   Check if this matches your expected training timeline.")
    
    elif len(existing_files) >= 4:  # At least core files exist
        print("⚠️  Most VFL model files are present")
        print("🔍 VERDICT: VFL training possibly succeeded with some issues")
        print("   Core model files exist but some auxiliary files may be missing.")
        
    else:
        print("❌ Multiple VFL model files are missing")
        print("💥 VERDICT: VFL training likely FAILED")
        print("   Insufficient model files to indicate successful training.")
    
    return len(existing_files), recent_files

def check_vfl_logs():
    """Check recent VFL training logs for success indicators"""
    
    print("\n" + "=" * 50)
    print("📋 Checking Recent VFL Training Logs")
    print("=" * 50)
    
    logs_dir = 'VFLClientModels/logs'
    if not os.path.exists(logs_dir):
        print("❌ Logs directory not found:", logs_dir)
        return
    
    # Look for recent log files
    log_files = []
    for file in os.listdir(logs_dir):
        if file.endswith('.log'):
            file_path = os.path.join(logs_dir, file)
            file_mtime = os.path.getmtime(file_path)
            log_files.append((file, file_path, file_mtime))
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x[2], reverse=True)
    
    print(f"📁 Found {len(log_files)} log files in {logs_dir}")
    
    # Show recent log files
    current_time = time.time()
    recent_logs = [lf for lf in log_files if (current_time - lf[2]) / 3600 < 24]  # Last 24 hours
    
    if recent_logs:
        print(f"🕒 Recent log files (last 24 hours):")
        for file, path, mtime in recent_logs[:5]:  # Show top 5 recent logs
            hours_ago = (current_time - mtime) / 3600
            modified_time = datetime.fromtimestamp(mtime)
            print(f"   📄 {file}")
            print(f"       Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours_ago:.1f} hours ago)")
    else:
        print("📅 No recent log files found in the last 24 hours")
    
    # Look for VFL-specific success indicators in recent logs
    success_indicators = [
        "Final model saved to VFLClientModels/saved_models/vfl_automl_xgboost_simple_model.keras",
        "AutoML search completed. Best model architecture found.",
        "Best hyperparameters saved to VFLClientModels/saved_models/best_hyperparameters_homoenc_dp.pkl"
    ]
    
    print(f"\n🔍 Searching for VFL success indicators in recent logs...")
    
    indicators_found = []
    for file, path, mtime in recent_logs[:3]:  # Check top 3 recent logs
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for indicator in success_indicators:
                    if indicator in content:
                        indicators_found.append((file, indicator))
        except Exception as e:
            print(f"⚠️  Could not read {file}: {e}")
    
    if indicators_found:
        print("✅ VFL success indicators found in logs:")
        for file, indicator in indicators_found:
            print(f"   📄 {file}:")
            print(f"      '{indicator[:80]}{'...' if len(indicator) > 80 else ''}'")
        print()
        print("🎉 LOG VERDICT: VFL training SUCCESS confirmed by log analysis")
    else:
        print("❌ No VFL success indicators found in recent logs")
        print("📝 LOG VERDICT: Logs do not confirm VFL training success")

def main():
    """Main verification function"""
    print("🚀 VFL Training Success Verification Tool")
    print("=" * 60)
    print("This tool helps verify if VFL training actually succeeded,")
    print("even when automated pipeline reports failure.")
    print()
    
    # Check model files
    file_count, recent_files = check_vfl_model_files()
    
    # Check logs
    check_vfl_logs()
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("🎯 FINAL RECOMMENDATION")
    print("=" * 60)
    
    if file_count >= 6 and recent_files:
        print("✅ VFL training appears to have SUCCEEDED")
        print("   - All critical model files are present")
        print("   - Files have recent modification timestamps")
        print("   - This suggests the automated pipeline had a false negative")
        print()
        print("🔧 Next steps:")
        print("   1. The enhanced pipeline logic should now correctly detect this success")
        print("   2. You can manually continue with explanation model updates if needed")
        print("   3. Monitor future runs with the improved success detection")
        
    elif file_count >= 4:
        print("⚠️  VFL training results are UNCLEAR")
        print("   - Some model files are present but verification is incomplete")
        print("   - Manual inspection of model quality is recommended")
        print()
        print("🔧 Next steps:")
        print("   1. Test the VFL models manually to verify functionality")
        print("   2. Check VFL training logs for detailed error messages")
        print("   3. Consider re-running VFL training if models are corrupted")
        
    else:
        print("❌ VFL training likely FAILED")
        print("   - Critical model files are missing")
        print("   - Automated pipeline failure detection appears correct")
        print()
        print("🔧 Next steps:")
        print("   1. Review detailed error logs from the automated pipeline")
        print("   2. Check system resources (memory, disk space)")
        print("   3. Verify all dependencies are installed")
        print("   4. Consider increasing timeout if process was killed")

if __name__ == "__main__":
    main()