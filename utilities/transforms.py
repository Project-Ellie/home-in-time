import apache_beam as beam


class MapAndFilterErrors(beam.PTransform):
  """Like beam.Map but filters out erros in the map_fn."""

  class _MapAndFilterErrorsDoFn(beam.DoFn):
    """Count the bad examples using a beam metric."""

    def __init__(self, fn):
      self._fn = fn
      # Create a counter to measure number of bad elements.
      self._bad_elements_counter = beam.metrics.Metrics.counter(
          'flight_data', 'bad_elements')

    def process(self, element):
      try:
        yield self._fn(element)
      except Exception:  # pylint: disable=broad-except
        # Catch any exception the above call.
        self._bad_elements_counter.inc(1)

  def __init__(self, fn):
    self._fn = fn

  def expand(self, pcoll):
    return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))
