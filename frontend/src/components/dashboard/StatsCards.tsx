import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { DashboardStats } from '../../services/types';

interface StatsCardsProps {
  stats: DashboardStats;
}

const StatsCards: React.FC<StatsCardsProps> = ({ stats }) => {
  const statCards = [
    {
      title: 'Total Pipeline Runs',
      value: stats.total_runs,
      color: '#1976d2',
      icon: '📊'
    },
    {
      title: 'Success Rate',
      value: `${stats.success_rate}%`,
      color: '#388e3c',
      icon: '✅'
    },
    {
      title: 'Last 24h Runs',
      value: stats.recent_runs_24h,
      color: '#f57c00',
      icon: '⏰'
    },
    {
      title: 'Avg Duration (s)',
      value: stats.average_duration,
      color: '#7b1fa2',
      icon: '⏱️'
    }
  ];

  return (
    <Box sx={{ 
      display: 'grid', 
      gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
      gap: 3,
      mb: 3 
    }}>
      {statCards.map((card, index) => (
        <Box key={index}>
          <Card 
            sx={{ 
              height: '100%',
              background: `linear-gradient(135deg, ${card.color}15, ${card.color}05)`,
              border: `1px solid ${card.color}30`,
              transition: 'transform 0.2s ease-in-out',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: `0 8px 25px ${card.color}20`
              }
            }}
          >
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h4" component="div" sx={{ fontWeight: 'bold', color: card.color }}>
                    {card.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    {card.title}
                  </Typography>
                </Box>
                <Typography variant="h3" sx={{ opacity: 0.3 }}>
                  {card.icon}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      ))}
    </Box>
  );
};

export default StatsCards;
