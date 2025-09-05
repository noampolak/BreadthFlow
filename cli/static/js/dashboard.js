async function loadData() {
    try {
        // Load summary data
        const summaryResponse = await fetch('/api/summary');
        const summary = await summaryResponse.json();
        
        if (summary.error) {
            console.error('Summary error:', summary.error);
            document.getElementById('total-runs').textContent = 'Error';
            return;
        }
        
        document.getElementById('total-runs').textContent = summary.total_runs || 0;
        document.getElementById('success-rate').textContent = (summary.success_rate || 0) + '%';
        document.getElementById('recent-runs').textContent = summary.recent_runs || 0;
        document.getElementById('avg-duration').textContent = (summary.avg_duration || 0).toFixed(1);
        
        // Load runs data with pagination
        const currentPage = window.currentPage || 1;
        const runsResponse = await fetch(`/api/runs?page=${currentPage}&per_page=10`);
        const runsData = await runsResponse.json();
        
        const tbody = document.getElementById('runs-tbody');
        tbody.innerHTML = '';
        
        if (runsData.error) {
            tbody.innerHTML = '<tr><td colspan="5">Error loading runs: ' + runsData.error + '</td></tr>';
            return;
        }
        
        const runs = runsData.runs || [];
        if (!runs.length) {
            tbody.innerHTML = '<tr><td colspan="5">No pipeline runs found. Run a demo to see data!</td></tr>';
            return;
        }
        
        runs.forEach(run => {
            const row = document.createElement('tr');
            const startTime = new Date(run.start_time).toLocaleString();
            const command = run.command.length > 30 ? run.command.substring(0, 30) + '...' : run.command;
            const runId = run.run_id.substring(0, 8);
            
            row.innerHTML = `
                <td>${command}</td>
                <td><span class="status-${run.status}">${run.status}</span></td>
                <td>${(run.duration || 0).toFixed(1)}s</td>
                <td>${startTime}</td>
                <td>
                    <button class="details-btn" onclick="showRunDetails('${run.run_id}')">Details</button>
                    ${run.command.includes('signals') || run.command.includes('backtest') ? 
                      `<button class="export-btn" onclick="exportRunData('${run.run_id}')">Export</button>` : ''}
                </td>
            `;
            tbody.appendChild(row);
        });
        
        // Update pagination controls
        if (runsData.pagination) {
            updatePagination(runsData.pagination);
        }
        
        document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleString();
        
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('total-runs').textContent = 'Error';
    }
}

function updatePagination(pagination) {
    const container = document.getElementById('pagination-container');
    const info = document.getElementById('pagination-info');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    const pageNumbers = document.getElementById('page-numbers');
    
    if (pagination.total_count === 0) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'block';
    
    // Update info
    const start = (pagination.page - 1) * pagination.per_page + 1;
    const end = Math.min(pagination.page * pagination.per_page, pagination.total_count);
    info.textContent = `Showing ${start}-${end} of ${pagination.total_count} runs`;
    
    // Update buttons
    prevBtn.disabled = !pagination.has_prev;
    nextBtn.disabled = !pagination.has_next;
    
    // Update page numbers
    let pageHtml = '';
    const maxPages = 5;
    let startPage = Math.max(1, pagination.page - Math.floor(maxPages / 2));
    let endPage = Math.min(pagination.total_pages, startPage + maxPages - 1);
    
    if (endPage - startPage + 1 < maxPages) {
        startPage = Math.max(1, endPage - maxPages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        if (i === pagination.page) {
            pageHtml += `<span class="page-number active">${i}</span>`;
        } else {
            pageHtml += `<span class="page-number" onclick="goToPage(${i})">${i}</span>`;
        }
    }
    
    pageNumbers.innerHTML = pageHtml;
}

function changePage(delta) {
    const currentPage = window.currentPage || 1;
    window.currentPage = Math.max(1, currentPage + delta);
    loadData();
}

function goToPage(page) {
    window.currentPage = page;
    loadData();
}

// Load data on page load
loadData();

// Auto-refresh every 30 seconds
setInterval(loadData, 30000);

// Modal functions
function showRunDetails(runId) {
    document.getElementById('runModal').style.display = 'block';
    document.getElementById('modalContent').innerHTML = 'Loading run details...';
    
    fetch(`/api/run/${runId}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('modalContent').innerHTML = formatRunDetails(data);
        })
        .catch(error => {
            document.getElementById('modalContent').innerHTML = `<p>Error loading details: ${error}</p>`;
        });
}

function closeModal() {
    document.getElementById('runModal').style.display = 'none';
}

function formatRunDetails(data) {
    if (data.error) {
        return `<p>Error: ${data.error}</p>`;
    }
    
    let html = `
        <h2>Run Details</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div>
                <h3>Basic Information</h3>
                <p><strong>Run ID:</strong> ${data.run_id}</p>
                <p><strong>Command:</strong> ${data.command}</p>
                <p><strong>Status:</strong> <span class="status-${data.status}">${data.status}</span></p>
                <p><strong>Duration:</strong> ${data.duration ? data.duration.toFixed(2) + 's' : 'N/A'}</p>
                ${data.metadata && data.metadata.timeframe ? `<p><strong>Timeframe:</strong> ${data.metadata.timeframe}</p>` : ''}
            </div>
            <div>
                <h3>Timing</h3>
                <p><strong>Started:</strong> ${new Date(data.start_time).toLocaleString()}</p>
                <p><strong>Ended:</strong> ${data.end_time ? new Date(data.end_time).toLocaleString() : 'N/A'}</p>
                ${data.metadata && data.metadata.symbols ? `<p><strong>Symbols:</strong> ${Array.isArray(data.metadata.symbols) ? data.metadata.symbols.join(', ') : data.metadata.symbols}</p>` : ''}
                ${data.error_message ? `<p><strong>Error:</strong> ${data.error_message}</p>` : ''}
            </div>
        </div>
    `;
    
    // Add signal-specific information if available
    if (data.command.includes('signals') && data.signals_data) {
        html += formatSignalsData(data.signals_data);
    }
    
    // Add backtest-specific information if available
    if (data.command.includes('backtest') && data.backtest_data) {
        html += formatBacktestData(data.backtest_data);
    }
    
    return html;
}

function formatSignalsData(signals) {
    return `
        <div style="margin-top: 20px;">
            <h3>Signal Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${signals.total_signals || 0}</div>
                    <div class="stat-label">Total Signals</div>
                </div>
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${signals.buy_signals || 0}</div>
                    <div class="stat-label">Buy Signals</div>
                </div>
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${signals.sell_signals || 0}</div>
                    <div class="stat-label">Sell Signals</div>
                </div>
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${signals.avg_confidence || 0}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
        </div>
    `;
}

function formatBacktestData(backtest) {
    return `
        <div style="margin-top: 20px;">
            <h3>Backtest Results</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${(backtest.total_return * 100).toFixed(1)}%</div>
                    <div class="stat-label">Total Return</div>
                </div>
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${backtest.sharpe_ratio || 0}</div>
                    <div class="stat-label">Sharpe Ratio</div>
                </div>
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${(backtest.max_drawdown * 100).toFixed(1)}%</div>
                    <div class="stat-label">Max Drawdown</div>
                </div>
                <div class="stat-card" style="margin: 0;">
                    <div class="stat-value">${backtest.total_trades || 0}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
            </div>
        </div>
    `;
}

function exportRunData(runId) {
    window.open(`/api/signals/export?run_id=${runId}`, '_blank');
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('runModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}
