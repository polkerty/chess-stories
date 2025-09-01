let gameId = null;
let positions = [];
let analyses = {};
let board = null;

function winnerLabelToScalar(label) {
  if (label === "White") return 1;
  if (label === "Draw") return 0.5;
  if (label === "Black") return 0;
  return null;
}

const ctx = document.getElementById('predictionChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      data: [],
      borderColor: '#00ff99',
      pointBackgroundColor: '#00ff99',
      tension: 0,
    }]
  },
  options: {
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: {
          callback: function(val){
            if(val === 1) return 'White';
            if(val === 0.5) return 'Draw';
            if(val === 0) return 'Black';
            return '';
          }
        }
      }
    },
    plugins: {
      legend: { display: false }
    },
    onClick: (evt, elements) => {
      if(elements.length > 0){
        const idx = elements[0].index;
        showPosition(idx);
      }
    }
  }
});

async function showPosition(idx){
  const pos = positions[idx];
  board.position(pos.fen);
  document.getElementById('metadata').textContent = `Ply ${pos.ply} - ${pos.move}`;
  if(analyses[pos.ply]){
    renderAnalysis(idx, analyses[pos.ply]);
  } else {
    document.getElementById('summary').textContent = 'Analyzing...';
    const resp = await fetch(`/api/game/${gameId}/analysis/${pos.ply}`);
    const data = await resp.json();
    analyses[pos.ply] = data;
    renderAnalysis(idx, data);
  }
}

function renderAnalysis(idx, data){
  document.getElementById('summary').textContent = data.analysis_json ? data.analysis_json.comment : data.analysis_raw;
  const scalar = winnerLabelToScalar(data.analysis_json ? data.analysis_json.winner_pred : null);
  chart.data.datasets[0].data[idx] = scalar;
  chart.update();
}

const form = document.getElementById('loadForm');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const url = document.getElementById('gameUrl').value.trim();
  const resp = await fetch('/api/game', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({url})
  });
  const data = await resp.json();
  gameId = data.game_id;
  positions = data.positions;
  analyses = {};
  chart.data.labels = positions.map(p => p.ply);
  chart.data.datasets[0].data = positions.map(_ => null);
  chart.update();
  if(!board){
    board = Chessboard('board', positions[0].fen);
  } else {
    board.position(positions[0].fen);
  }
  showPosition(0);
});
