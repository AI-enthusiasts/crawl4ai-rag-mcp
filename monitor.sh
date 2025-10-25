#!/bin/bash
CONTAINER="mcp-crawl4ai-ns4gso0k8co8kc40c0k4s8ok-225650803864"
DURATION=120
INTERVAL=5

echo "=== Monitoring for 2 minutes ==="
echo "Time | Processes | Memory | CPU | Activity"
echo "-----+-----------+--------+-----+---------"

START=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    [ $ELAPSED -ge $DURATION ] && break
    
    PROCS=$(docker exec $CONTAINER ps aux 2>/dev/null | grep chromium | wc -l)
    MEM=$(docker stats --no-stream --format "{{.MemUsage}}" $CONTAINER 2>/dev/null | cut -d'/' -f1)
    CPU=$(docker stats --no-stream --format "{{.CPUPerc}}" $CONTAINER 2>/dev/null)
    ACTIVITY=$(docker logs --tail 3 $CONTAINER 2>&1 | grep -oE "crawl_batch|arun_many|Crawling complete|CallToolRequest" | tail -1)
    
    printf "%02d:%02d | %9s | %6s | %3s | %s\n" $((ELAPSED/60)) $((ELAPSED%60)) "$PROCS" "$MEM" "$CPU" "$ACTIVITY"
    sleep $INTERVAL
done

echo ""
echo "=== Final Summary ==="
docker exec $CONTAINER ps aux | grep chromium | wc -l | xargs echo "Chromium processes:"
docker logs $CONTAINER 2>&1 | grep "Initializing AsyncWebCrawler" | wc -l | xargs echo "Crawler initializations:"
docker logs $CONTAINER 2>&1 | grep -E "crawl_batch|arun_many" | wc -l | xargs echo "Crawl operations:"
