"""
Test the scheduler locally before deploying to Railway
Run this to verify everything works
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'DAGS', '.env')
load_dotenv(env_path)

print("\n" + "="*80)
print("TESTING SCHEDULER LOCALLY")
print("="*80 + "\n")

# Check environment variables
print("[1/3] Checking environment variables...")
required_vars = ['ARANGO_HOST', 'ARANGO_DATABASE', 'ARANGO_USERNAME', 'ARANGO_PASSWORD']
for var in required_vars:
    value = os.getenv(var)
    if value:
        display = '***' if 'PASSWORD' in var else value
        print(f"  [OK] {var}: {display}")
    else:
        print(f"  [FAIL] {var}: NOT SET")
        sys.exit(1)

# Import the scheduler
print("\n[2/3] Importing scheduler app...")
try:
    from app import run_pipeline
    print("  [OK] Scheduler imported successfully")
except Exception as e:
    print(f"  [FAIL] Failed to import: {e}")
    sys.exit(1)

# Run pipeline once
print("\n[3/3] Running pipeline once (this will take ~3-5 minutes)...")
print("-"*80)

try:
    success = run_pipeline()

    if success:
        print("\n" + "="*80)
        print("[OK] SCHEDULER TEST PASSED")
        print("="*80)
        print("\nThe scheduler is working correctly!")
        print("You can now deploy it to Railway.")
        print("\nNext steps:")
        print("1. git add scheduler/")
        print("2. git commit -m 'Add Railway scheduler'")
        print("3. git push origin main")
        print("4. Deploy on Railway (see RAILWAY_DEPLOY.md)")
    else:
        print("\n" + "="*80)
        print("[FAIL] SCHEDULER TEST FAILED")
        print("="*80)
        print("\nCheck the errors above and fix before deploying.")

except KeyboardInterrupt:
    print("\n\n→ Test interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n[FAIL] Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
