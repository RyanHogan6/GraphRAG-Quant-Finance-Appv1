# 🚂 Railway Scheduler Deployment Guide

Deploy the Polymarket pipeline scheduler to Railway - runs your ETL pipeline every 10 minutes automatically, 24/7.

---

## 📋 Prerequisites

- ✅ GitHub repo with this code pushed
- ✅ Railway account (railway.app)
- ✅ ArangoDB Cloud credentials
- ✅ Existing Railway project (with your FastAPI backend)

---

## 🚀 Deployment Steps

### **Step 1: Push Code to GitHub**

```bash
# From the streamlit directory
git add scheduler/
git commit -m "Add Railway scheduler service"
git push origin main
```

### **Step 2: Add Service to Railway**

1. Go to **Railway Dashboard** (railway.app)
2. Open your **existing project** (where your backend is)
3. Click the **"+ New"** button (top right)
4. Select **"GitHub Repo"**
5. Choose your repository
6. Select **"Deploy from Subdirectory"**
7. Enter: `scheduler`
8. Click **"Deploy"**

### **Step 3: Configure Environment Variables**

**RECOMMENDED: Reference your existing backend's variables** (they're already set!):

1. Click on the **scheduler service** in Railway
2. Go to **"Variables"** tab
3. Click **"+ Reference Variable"** button
4. Select your **backend service** from the dropdown
5. Add references for:
   - `ARANGO_HOST`
   - `ARANGO_DATABASE`
   - `ARANGO_USERNAME`
   - `ARANGO_PASSWORD`
   - `ARANGO_GRAPH_NAME`

**This way:**
- ✅ No duplicate credentials to manage
- ✅ Update once, applies everywhere
- ✅ More secure (credentials stay in one place)

### **Step 4: Deploy & Monitor**

Railway will automatically:
- ✅ Build the service
- ✅ Install dependencies
- ✅ Start the scheduler
- ✅ Begin running pipeline every 10 minutes

**Monitor deployment:**
- Click **"Deployments"** tab to see build progress
- Click **"Logs"** tab to see execution logs
- You should see: "PIPELINE EXECUTION STARTED" every 10 minutes

---

## 📊 What Gets Deployed

```
Railway Project
├── Service 1: FastAPI Backend (existing)
│   └── Serves API on port 8000
└── Service 2: Pipeline Scheduler (new) ✨
    └── Runs ETL every 10 minutes
```

**Both services:**
- Share the same environment variables
- Connect to the same ArangoDB
- Are in the same Railway project
- Use the same billing

---

## 🔍 Verify It's Working

### **Check Logs in Railway:**

You should see logs like:
```
================================================================================
PIPELINE EXECUTION STARTED: 2026-01-12 04:00:00
================================================================================
[1/7] Fetching markets from Polymarket API...
✓ Fetched 24,031 markets
[2/7] Skipping trader fetch (not 6-hour cycle)
[3/7] Engineering features...
✓ Engineered features for 24,031 markets
[4/7] Connecting to ArangoDB...
✓ Connected to database
[5/7] Uploading markets to ArangoDB...
✓ Markets - Inserted: 0, Updated: 24,031, Errors: 0
[6/7] Saving price history snapshots...
✓ Price snapshots - Inserted: 23,500, Errors: 0
[7/7] Skipping edge building (not 6-hour cycle)
================================================================================
PIPELINE COMPLETED SUCCESSFULLY
Duration: 187.3 seconds
Markets: 0 inserted, 24,031 updated
Price snapshots: 23,500
================================================================================
```

### **Check Database:**

Query your ArangoDB to verify:
```aql
// Check latest price snapshots
FOR doc IN polymarket_price_history
    SORT doc.timestamp DESC
    LIMIT 10
    RETURN {
        market_id: doc.market_id,
        datetime: doc.datetime,
        yes_price: doc.yes_price
    }

// Should show fresh data from last 10 minutes!
```

---

## ⏰ Execution Schedule

| Frequency | What Runs | Duration |
|-----------|-----------|----------|
| **Every 10 min** | Markets + price snapshots | ~2-3 min |
| **Every 6 hours** | + Traders + graph edges | ~5-7 min |
| **Daily at 00:00** | + Cleanup old data | +30 sec |

**Expected data growth:**
- ~144 executions per day
- ~23,000 price snapshots per execution
- ~3.3M snapshots per day
- Auto-cleanup keeps it at 90 days = ~300M snapshots

---

## 💰 Cost Estimate

Railway charges for:
- **Compute:** ~$5-10/month (always running)
- **Network:** ~$1-2/month (API calls)

**Total:** ~$6-12/month

**Much cheaper than:**
- Google Cloud Composer: ~$300/month
- AWS MWAA: ~$350/month
- Running Docker 24/7 on your PC: Electricity costs + wear

---

## 🛠️ Troubleshooting

### **Service won't start:**
- Check **Logs** tab for errors
- Verify environment variables are set
- Ensure `ARANGO_PASSWORD` is correct

### **Pipeline not running:**
- Check logs for "PIPELINE EXECUTION STARTED"
- Verify it appears every 10 minutes
- Check for error messages

### **No price snapshots saved:**
- Verify `outcome_prices` parsing is working
- Check logs for "Price snapshots - Inserted: 0"
- Run test script locally first

### **Railway service keeps restarting:**
- Check if environment variables are missing
- Look for Python import errors in logs
- Verify `requirements.txt` has all dependencies

---

## 🔄 Updating the Scheduler

To update the scheduler code:

```bash
# Make changes to scheduler/app.py
git add scheduler/
git commit -m "Update scheduler logic"
git push origin main

# Railway auto-deploys on git push!
```

Railway will automatically:
1. Detect the git push
2. Rebuild the service
3. Deploy the new version
4. Resume running the scheduler

---

## 🎯 Monitoring & Maintenance

### **Daily Checks:**
- ✅ Check Railway logs for errors
- ✅ Verify price snapshots are growing
- ✅ Check ArangoDB collection sizes

### **Weekly Checks:**
- ✅ Review execution times (should be ~2-3 min)
- ✅ Check for failed executions
- ✅ Verify data freshness (< 10 min old)

### **Monthly Checks:**
- ✅ Review Railway billing
- ✅ Check database storage usage
- ✅ Verify cleanup job is running

---

## 📞 Support

**Issues?**
- Check Railway logs first
- Verify environment variables
- Test pipeline locally with `python app.py`
- Check ArangoDB connectivity

**Railway Documentation:**
- https://docs.railway.app/

---

## ✅ Success Checklist

After deployment, verify:

- [ ] Service shows "Active" in Railway dashboard
- [ ] Logs show "PIPELINE EXECUTION STARTED" every 10 minutes
- [ ] Price snapshots are increasing in ArangoDB
- [ ] Markets are being updated (check `updated_at` field)
- [ ] No error messages in logs
- [ ] Health checks appear every 1 minute

---

**🎉 Once deployed, your pipeline runs autonomously 24/7!**

Your PC can be off, Docker can be closed - Railway keeps it running.
