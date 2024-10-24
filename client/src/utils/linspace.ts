export const linspace = (start: number, stop: number, num: number) => {
  const arr = [];
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    arr.push(Math.round(start + (step * i)));
  }
  return arr;
};