{% extends 'full.tpl'%}
{% block any_cell %}
{% if cell['metadata'].get('extensions', {}).get('jupyter_dashboards', {}).get('views', {}).get('report_default', {}).get('hidden', false) %}

{% else %}
{{ super()}}

{% endif %}
{% endblock any_cell %}

{% block input_group %}
{% endblock input_group %}`
