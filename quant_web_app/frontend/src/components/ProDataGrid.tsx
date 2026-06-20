import React, { useState, useMemo } from 'react';
import { Sparklines, SparklinesLine } from 'react-sparklines';
import { useApp } from '../context/AppContext';

interface ProDataGridProps {
  data?: any[];
  columns?: {
    key: string;
    label: string;
    format?: (value: any) => string;
    align?: 'left' | 'center' | 'right';
    sortable?: boolean;
    type?: 'number' | 'string' | 'percent' | 'rank' | 'sparkline' | 'indicator';
  }[];
  onRowClick?: (row: any) => void;
  title?: string;
  subtitle?: string;
}

const ProDataGrid: React.FC<ProDataGridProps> = ({ data = [], columns = [], onRowClick, title, subtitle }) => {
  const { theme } = useApp();
  const isDark = theme === 'dark';

  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  // 防御：如果 columns 为空或数据为空，显示提示
  if (!columns || columns.length === 0 || !data || data.length === 0) {
    return (
      <div style={{
        background: isDark ? 'rgba(17,24,39,0.6)' : 'rgba(255,255,255,0.8)',
        borderRadius: '12px',
        padding: '20px',
        border: `1px solid ${isDark ? '#1F2937' : '#E5E7EB'}`,
        textAlign: 'center',
        color: isDark ? '#6B7280' : '#4B5563',
      }}>
        {title && <h4 style={{ marginBottom: '8px' }}>{title}</h4>}
        <span>暂无数据</span>
      </div>
    );
  }

  const sortedData = useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => {
      const aVal = a[sortKey] ?? 0;
      const bVal = b[sortKey] ?? 0;
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      return sortDirection === 'asc'
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
  }, [data, sortKey, sortDirection]);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  const getRankBadge = (index: number) => {
    if (index === 0) return <span style={{ fontSize: '16px' }}>🥇</span>;
    if (index === 1) return <span style={{ fontSize: '16px' }}>🥈</span>;
    if (index === 2) return <span style={{ fontSize: '16px' }}>🥉</span>;
    return <span style={{ color: isDark ? '#6B7280' : '#4B5563', fontSize: '12px' }}>#{index + 1}</span>;
  };

  const getIndicator = (value: number) => {
    if (value > 0) return <span style={{ color: '#00F5A0' }}>▲</span>;
    if (value < 0) return <span style={{ color: '#FF0050' }}>▼</span>;
    return <span style={{ color: '#F59E0B' }}>—</span>;
  };

  const formatValue = (value: any, col: any) => {
    if (value === undefined || value === null) return '—';
    if (col.format) return col.format(value);
    if (col.type === 'percent') return `${(value * 100).toFixed(2)}%`;
    if (col.type === 'number') return typeof value === 'number' ? value.toFixed(3) : value;
    return String(value);
  };

  const getDataBar = (value: number, max: number) => {
    const pct = Math.abs(value / max) * 100;
    return {
      width: Math.min(pct, 100),
      color: value >= 0 ? 'rgba(0, 245, 160, 0.25)' : 'rgba(255, 0, 80, 0.25)',
      barColor: value >= 0 ? '#00F5A0' : '#FF0050',
    };
  };

  // 计算各列最大值用于数据条
  const maxValues: Record<string, number> = {};
  columns.forEach(col => {
    if (col.type === 'number' || col.type === 'percent') {
      const vals = data.map(d => Math.abs(d[col.key] || 0));
      maxValues[col.key] = Math.max(...vals, 0.001);
    }
  });

  const bg = isDark ? 'rgba(10,14,23,0.85)' : 'rgba(255,255,255,0.85)';
  const border = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
  const textPrimary = isDark ? '#E0E0E0' : '#111827';
  const textSecondary = isDark ? '#6B7280' : '#4B5563';
  const headerBg = isDark ? 'rgba(10,14,23,0.95)' : 'rgba(255,255,255,0.95)';

  return (
    <div style={{
      background: bg,
      backdropFilter: 'blur(20px)',
      border: `1px solid ${border}`,
      borderRadius: '16px',
      padding: '20px',
      boxShadow: isDark ? '0 20px 60px rgba(0,0,0,0.6)' : '0 20px 60px rgba(0,0,0,0.08)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h3 style={{ color: textPrimary, fontSize: '14px', fontWeight: 600, margin: 0 }}>
            {title || '📊 数据网格'}
          </h3>
          {subtitle && (
            <span style={{ color: textSecondary, fontSize: '11px', marginTop: '2px' }}>{subtitle}</span>
          )}
        </div>
        <div style={{ display: 'flex', gap: '12px', fontSize: '10px', color: textSecondary }}>
          <span>📦 {data.length} 条</span>
          <span>🔄 点击表头排序</span>
        </div>
      </div>

      <div style={{ overflow: 'auto', maxHeight: '420px' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '12px',
        }}>
          <thead style={{
            position: 'sticky',
            top: 0,
            zIndex: 10,
          }}>
            <tr>
              <th style={{
                padding: '8px 10px',
                textAlign: 'center',
                color: textSecondary,
                fontWeight: 600,
                fontSize: '10px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}`,
                background: headerBg,
              }}>#</th>
              {columns.map(col => (
                <th
                  key={col.key}
                  onClick={() => col.sortable !== false && handleSort(col.key)}
                  style={{
                    padding: '8px 10px',
                    textAlign: col.align || 'left',
                    color: sortKey === col.key ? '#00E5FF' : textSecondary,
                    fontWeight: sortKey === col.key ? 600 : 500,
                    fontSize: '10px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}`,
                    background: headerBg,
                    cursor: col.sortable !== false ? 'pointer' : 'default',
                    transition: 'color 0.2s',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span style={{ marginLeft: '4px' }}>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedData.map((row, index) => {
              const isHovered = hoveredRow === index;
              const firstCol = columns[0];
              const firstVal = row[firstCol?.key] || 0;
              const maxVal = maxValues[firstCol?.key] || 1;

              return (
                <tr
                  key={index}
                  onClick={() => onRowClick?.(row)}
                  onMouseEnter={() => setHoveredRow(index)}
                  onMouseLeave={() => setHoveredRow(null)}
                  style={{
                    cursor: onRowClick ? 'pointer' : 'default',
                    transition: 'all 0.2s ease',
                    background: isHovered ? (isDark ? 'rgba(0,229,255,0.04)' : 'rgba(0,229,255,0.04)') : 'transparent',
                    borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)'}`,
                  }}
                >
                  <td style={{
                    padding: '8px 10px',
                    textAlign: 'center',
                    color: textSecondary,
                    fontSize: '11px',
                  }}>
                    {getRankBadge(index)}
                  </td>
                  {columns.map(col => {
                    const value = row[col.key];
                    const isNumber = typeof value === 'number';
                    const isPercent = col.type === 'percent';
                    const maxVal = maxValues[col.key] || 1;

                    let content;
                    if (col.type === 'sparkline') {
                      content = (
                        <Sparklines data={row[col.key] || [0, 1, 2, 3, 2, 4, 3]} width={60} height={24}>
                          <SparklinesLine
                            color={index < 3 ? '#00E5FF' : '#6B7280'}
                            style={{ strokeWidth: 1.5, fill: 'none' }}
                          />
                        </Sparklines>
                      );
                    } else if (col.type === 'indicator') {
                      content = (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          {getIndicator(value)}
                          <span style={{ color: value > 0 ? '#00F5A0' : value < 0 ? '#FF0050' : '#F59E0B' }}>
                            {formatValue(value, col)}
                          </span>
                        </span>
                      );
                    } else if ((col.type === 'number' || col.type === 'percent') && isNumber && value !== 0) {
                      const bar = getDataBar(value, maxVal);
                      content = (
                        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            height: '100%',
                            width: `${bar.width}%`,
                            background: bar.color,
                            borderRadius: '4px',
                            transition: 'width 0.6s ease',
                            opacity: 0.3,
                          }} />
                          <span style={{
                            position: 'relative',
                            zIndex: 1,
                            color: value > 0 ? '#00F5A0' : '#FF0050',
                            fontWeight: 500,
                          }}>
                            {formatValue(value, col)}
                          </span>
                        </div>
                      );
                    } else {
                      content = formatValue(value, col);
                    }

                    return (
                      <td
                        key={col.key}
                        style={{
                          padding: '8px 10px',
                          textAlign: col.align || 'left',
                          color: textPrimary,
                          fontSize: '12px',
                          whiteSpace: 'nowrap',
                          position: 'relative',
                        }}
                      >
                        {content}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{
        marginTop: '12px',
        paddingTop: '12px',
        borderTop: `1px solid ${isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)'}`,
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: textSecondary,
      }}>
        <span>🏆 金/银/铜 = 夏普排名前三</span>
        <span>📊 数据条显示相对强弱</span>
      </div>

      <div style={{
        position: 'absolute',
        top: '-100px',
        right: '-100px',
        width: '400px',
        height: '400px',
        background: isDark ? 'radial-gradient(circle, rgba(0,229,255,0.03) 0%, transparent 70%)' : 'radial-gradient(circle, rgba(0,229,255,0.05) 0%, transparent 70%)',
        pointerEvents: 'none',
        borderRadius: '50%',
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-100px',
        left: '-100px',
        width: '300px',
        height: '300px',
        background: isDark ? 'radial-gradient(circle, rgba(139,92,246,0.03) 0%, transparent 70%)' : 'radial-gradient(circle, rgba(139,92,246,0.05) 0%, transparent 70%)',
        pointerEvents: 'none',
        borderRadius: '50%',
      }} />
    </div>
  );
};

export default ProDataGrid;