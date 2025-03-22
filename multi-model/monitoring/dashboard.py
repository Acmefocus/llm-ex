import grafana_api
from prometheus_client import start_http_server

class MonitoringDashboard:
    def __init__(self):
        start_http_server(8001)
        self.grafana = grafana_api.GrafanaAPI()
        
    def setup_dashboards(self):
        self.grafana.dashboard.import_dashboard({
            "dashboard": {
                "title": "Model Performance",
                "panels": [
                    {
                        "title": "API Calls",
                        "type": "graph",
                        "targets": [{
                            "expr": 'rate(successful_requests_total[5m])',
                            "legendFormat": "Success Rate"
                        }]
                    }
                ]
            }
        })
