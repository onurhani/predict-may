{% macro get_constant(constant_name) %}
    {{ var(constant_name) }}
{% endmacro %}

{#
Usage in models:
{{ get_constant('spi_scaling_factor') }}

This macro is a simple wrapper around var() for clarity
You could extend it to add validation or type checking
#}