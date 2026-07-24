# Third-Party Notices

## SOAP Optimizer

The file `src/greenonet/optimizers/soap.py` is derived from the official
preliminary SOAP implementation:

- Repository: https://github.com/nikhilvyas/SOAP
- Upstream commit: `a1e553530fde97d0e6b307d7c82ac6d38b072340`
- Upstream file SHA-256:
  `9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b`
- Copyright (c) 2024 Nikhil Vyas

The vendored file retains the upstream update equations and first-step
initialization behavior. Local modifications add strict argument and
dense-gradient validation, float64 model dtype bridges, type/lint isolation,
and read-only optimizer telemetry counters.

The upstream source is distributed under the MIT License:

```text
MIT License

Copyright (c) 2024 Nikhil Vyas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
