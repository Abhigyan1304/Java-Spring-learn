import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface ColumnSchema {
  id: string;
  label: string;
  type: 'text' | 'dropdown' | 'number';
  options?: string[];
}

export interface Row {
  [key: string]: string | boolean | number;
}

interface TableState {
  columns: any;
  schema: ColumnSchema[];
  rows: Row[];
  marketShare: number[];
}

const initialState: TableState = {
  columns: [],
  schema: [],
  rows: [],
  marketShare: [],
};

function randomMarketShare(n: number): number[] {
  // Generate n random numbers, normalize to sum 100
  const vals = Array.from({ length: n }, () => Math.random());
  const sum = vals.reduce((a, b) => a + b, 0);
  let shares = vals.map(v => Math.round((v / sum) * 100));
  // Adjust last value so total is exactly 100
  const diff = 100 - shares.reduce((a, b) => a + b, 0);
  shares[shares.length - 1] += diff;
  return shares;
}

const tableSlice = createSlice({
  name: 'table',
  initialState,
  reducers: {
    setSchema(state, action: PayloadAction<ColumnSchema[]>) {
      state.schema = action.payload;
    },
    setRows(state, action: PayloadAction<Row[]>) {
      state.rows = action.payload.map(row => ({
        ...row,
        price: typeof row.price === "string"
          ? parseInt(row.price.replace(/\D/g, ""), 10)
          : row.price
      }));
      state.marketShare = randomMarketShare(state.rows.length);
    },
    updateCell(
      state,
      action: PayloadAction<{ rowIndex: number; colId: string; value: string | boolean | number }>
    ) {
      const { rowIndex, colId, value } = action.payload;
      state.rows[rowIndex][colId] = value;
      // Do NOT update marketShare here!
    },
    addPhoneColumn(state, action: PayloadAction<Row>) {
      state.rows.push({ ...action.payload });
      state.marketShare = randomMarketShare(state.rows.length);
    },
    recalculateMarketShare(state) {
      state.marketShare = randomMarketShare(state.rows.length);
    }
  },
});

export const { setSchema, setRows, updateCell, addPhoneColumn, recalculateMarketShare } = tableSlice.actions;
export default tableSlice.reducer;
